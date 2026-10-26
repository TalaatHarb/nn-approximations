package net.talaatharb.nn.tracking;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Minimal client for the MLflow tracking REST API (2.x/3.x): experiments, runs, batched metrics/params/tags and
 * artifact upload through the tracking server's artifact proxy ({@code mlflow-artifacts:} URIs).
 */
public class MlflowClient {

	/** Request limits of the log-batch endpoint. */
	static final int MAX_METRICS_PER_BATCH = 1000;
	static final int MAX_PARAMS_OR_TAGS_PER_BATCH = 100;
	static final int MAX_PARAM_LENGTH = 6000;

	public record Metric(String key, double value, long timestamp, long step) {
	}

	public record RunInfo(String experimentId, String runId, String artifactUri) {
	}

	public static final class MlflowException extends IOException {
		private static final long serialVersionUID = 1L;
		private final int status;
		private final String errorCode;

		MlflowException(int status, String errorCode, String message) {
			super(message);
			this.status = status;
			this.errorCode = errorCode;
		}

		public int status() {
			return status;
		}

		public String errorCode() {
			return errorCode;
		}
	}

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final URI base;
	private final HttpClient http;
	private final String authorization;

	public MlflowClient(URI base) {
		this(base, HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10))
				.followRedirects(HttpClient.Redirect.NORMAL).build(), authorizationFromEnvironment());
	}

	MlflowClient(URI base, HttpClient http, String authorization) {
		final String text = base.toString();
		this.base = URI.create(text.endsWith("/") ? text.substring(0, text.length() - 1) : text);
		this.http = http;
		this.authorization = authorization;
	}

	public URI base() {
		return base;
	}

	/** The server's version string; a cheap reachability check. */
	public String version() throws IOException {
		final var response = send(request("/version").GET().build());
		return response.body().trim();
	}

	/** Id of the named experiment, creating it when missing. */
	public String experimentId(String name) throws IOException {
		try {
			return getJson("/api/2.0/mlflow/experiments/get-by-name?experiment_name="
					+ URLEncoder.encode(name, StandardCharsets.UTF_8)).path("experiment").path("experiment_id")
					.asText();
		} catch (MlflowException missing) {
			if (!"RESOURCE_DOES_NOT_EXIST".equals(missing.errorCode()) && missing.status() != 404) {
				throw missing;
			}
		}
		try {
			return postJson("/api/2.0/mlflow/experiments/create", Map.of("name", name)).path("experiment_id")
					.asText();
		} catch (MlflowException race) {
			if (!"RESOURCE_ALREADY_EXISTS".equals(race.errorCode())) {
				throw race;
			}
			return experimentId(name);
		}
	}

	public RunInfo createRun(String experimentId, String runName, long startTime, Map<String, String> tags)
			throws IOException {
		final var body = new LinkedHashMap<String, Object>();
		body.put("experiment_id", experimentId);
		body.put("run_name", runName);
		body.put("start_time", startTime);
		body.put("tags", keyValues(tags));
		final JsonNode info = postJson("/api/2.0/mlflow/runs/create", body).path("run").path("info");
		return new RunInfo(experimentId, info.path("run_id").asText(), info.path("artifact_uri").asText());
	}

	/** Logs everything, split into as many log-batch calls as the server limits require. */
	public void logBatch(String runId, List<Metric> metrics, Map<String, String> params, Map<String, String> tags)
			throws IOException {
		final var paramEntries = new ArrayList<>(params.entrySet());
		final var tagEntries = new ArrayList<>(tags.entrySet());
		for (int i = 0; i < Math.max(paramEntries.size(), tagEntries.size()); i += MAX_PARAMS_OR_TAGS_PER_BATCH) {
			final var batch = new LinkedHashMap<String, Object>();
			batch.put("run_id", runId);
			batch.put("params", keyValues(slice(paramEntries, i, MAX_PARAMS_OR_TAGS_PER_BATCH)));
			batch.put("tags", keyValues(slice(tagEntries, i, MAX_PARAMS_OR_TAGS_PER_BATCH)));
			postJson("/api/2.0/mlflow/runs/log-batch", batch);
		}
		final var finite = metrics.stream().filter(metric -> Double.isFinite(metric.value())).toList();
		for (int i = 0; i < finite.size(); i += MAX_METRICS_PER_BATCH) {
			final var rows = finite.subList(i, Math.min(finite.size(), i + MAX_METRICS_PER_BATCH)).stream()
					.map(metric -> Map.<String, Object>of("key", metric.key(), "value", metric.value(), "timestamp",
							metric.timestamp(), "step", metric.step()))
					.toList();
			postJson("/api/2.0/mlflow/runs/log-batch", Map.of("run_id", runId, "metrics", rows));
		}
	}

	/** Status is one of RUNNING, FINISHED, FAILED or KILLED. */
	public void updateRun(String runId, String status, long endTime) throws IOException {
		postJson("/api/2.0/mlflow/runs/update", Map.of("run_id", runId, "status", status, "end_time", endTime));
	}

	/**
	 * Uploads one artifact. Only servers that proxy artifacts ({@code mlflow server --serve-artifacts}, the default
	 * since MLflow 2) accept uploads over HTTP; other artifact stores raise an exception explaining why.
	 */
	public void uploadArtifact(RunInfo run, String path, byte[] content) throws IOException {
		final String prefix = "mlflow-artifacts:";
		if (!run.artifactUri().startsWith(prefix)) {
			throw new IOException("The server stores artifacts at " + run.artifactUri()
					+ "; only servers that proxy artifacts (mlflow-artifacts:) accept uploads from this app");
		}
		String location = run.artifactUri().substring(prefix.length());
		if (location.startsWith("//")) {
			final int slash = location.indexOf('/', 2);
			location = slash < 0 ? "" : location.substring(slash);
		}
		final var segments = new ArrayList<String>();
		for (String segment : (location + "/" + path).split("/")) {
			if (!segment.isEmpty()) {
				segments.add(URLEncoder.encode(segment, StandardCharsets.UTF_8).replace("+", "%20"));
			}
		}
		send(request("/api/2.0/mlflow-artifacts/artifacts/" + String.join("/", segments))
				.header("Content-Type", "application/octet-stream")
				.PUT(HttpRequest.BodyPublishers.ofByteArray(content)).build());
	}

	public String runUrl(RunInfo run) {
		return base + "/#/experiments/" + run.experimentId() + "/runs/" + run.runId();
	}

	private JsonNode getJson(String path) throws IOException {
		return MAPPER.readTree(send(request(path).GET().build()).body());
	}

	private JsonNode postJson(String path, Object body) throws IOException {
		final var request = request(path).header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body))).build();
		final String response = send(request).body();
		return response.isBlank() ? MAPPER.createObjectNode() : MAPPER.readTree(response);
	}

	private HttpRequest.Builder request(String path) {
		final var builder = HttpRequest.newBuilder(URI.create(base + path)).timeout(Duration.ofSeconds(30))
				.header("Accept", "application/json");
		if (authorization != null) {
			builder.header("Authorization", authorization);
		}
		return builder;
	}

	private HttpResponse<String> send(HttpRequest request) throws IOException {
		final HttpResponse<String> response;
		try {
			response = http.send(request, HttpResponse.BodyHandlers.ofString());
		} catch (InterruptedException exception) {
			Thread.currentThread().interrupt();
			throw new IOException("Interrupted while contacting MLflow", exception);
		}
		if (response.statusCode() / 100 != 2) {
			String code = null;
			String message = response.body();
			try {
				final JsonNode error = MAPPER.readTree(response.body());
				code = error.path("error_code").asText(null);
				message = error.path("message").asText(message);
			} catch (IOException notJson) {
				// keep the raw body
			}
			if (message != null && message.length() > 300) {
				message = message.substring(0, 300) + "...";
			}
			throw new MlflowException(response.statusCode(), code, "MLflow " + request.method() + " "
					+ request.uri().getPath() + " failed with HTTP " + response.statusCode() + ": " + message);
		}
		return response;
	}

	private static <T> List<T> slice(List<T> values, int from, int size) {
		return from >= values.size() ? List.of() : values.subList(from, Math.min(values.size(), from + size));
	}

	private static List<Map<String, String>> keyValues(Map<String, String> values) {
		return keyValues(new ArrayList<>(values.entrySet()));
	}

	private static List<Map<String, String>> keyValues(List<Map.Entry<String, String>> entries) {
		return entries.stream().map(entry -> Map.of("key", entry.getKey(), "value", truncate(entry.getValue())))
				.toList();
	}

	private static String truncate(String value) {
		final String text = value == null ? "" : value;
		return text.length() <= MAX_PARAM_LENGTH ? text : text.substring(0, MAX_PARAM_LENGTH - 3) + "...";
	}

	static String authorizationFromEnvironment() {
		final String token = System.getenv("MLFLOW_TRACKING_TOKEN");
		if (token != null && !token.isBlank()) {
			return "Bearer " + token.trim();
		}
		final String user = System.getenv("MLFLOW_TRACKING_USERNAME");
		final String password = System.getenv("MLFLOW_TRACKING_PASSWORD");
		if (user != null && !user.isBlank() && password != null) {
			return "Basic " + Base64.getEncoder()
					.encodeToString((user + ":" + password).getBytes(StandardCharsets.UTF_8));
		}
		return null;
	}
}
