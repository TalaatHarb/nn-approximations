package net.talaatharb.nn.tracking;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/** In-memory stand-in for the MLflow tracking REST API, enforcing the log-batch limits of the real server. */
public final class FakeMlflowServer implements AutoCloseable {

	public static final class Run {
		public final String id;
		public final String experimentId;
		public final String name;
		public volatile String status = "RUNNING";
		public final Map<String, String> params = new ConcurrentHashMap<>();
		public final Map<String, String> tags = new ConcurrentHashMap<>();
		public final List<JsonNode> metrics = Collections.synchronizedList(new ArrayList<>());

		Run(String id, String experimentId, String name) {
			this.id = id;
			this.experimentId = experimentId;
			this.name = name;
		}

		public List<JsonNode> metrics(String key) {
			synchronized (metrics) {
				return metrics.stream().filter(metric -> metric.path("key").asText().equals(key)).toList();
			}
		}
	}

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final HttpServer server;
	private final AtomicInteger ids = new AtomicInteger();
	public final Map<String, String> experiments = new ConcurrentHashMap<>();
	public final Map<String, Run> runs = Collections.synchronizedMap(new LinkedHashMap<>());
	public final Map<String, byte[]> artifacts = new ConcurrentHashMap<>();
	public final List<String> requests = Collections.synchronizedList(new ArrayList<>());
	public volatile int failMetricBatchesWithStatus;
	public volatile String artifactUriScheme = "mlflow-artifacts:";
	/** When set, every request must carry exactly this Authorization header (like {@code mlflow server --app-name basic-auth}). */
	public volatile String requiredAuthorization;

	public FakeMlflowServer() throws IOException {
		server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		server.createContext("/", this::handle);
		server.start();
	}

	public URI uri() {
		return URI.create("http://127.0.0.1:" + server.getAddress().getPort());
	}

	public List<Run> runs() {
		synchronized (runs) {
			return List.copyOf(runs.values());
		}
	}

	@Override
	public void close() {
		server.stop(0);
	}

	private void handle(HttpExchange exchange) throws IOException {
		final String path = exchange.getRequestURI().getPath();
		final String method = exchange.getRequestMethod();
		requests.add(method + " " + path);
		final byte[] body = exchange.getRequestBody().readAllBytes();
		try {
			final String required = requiredAuthorization;
			if (required != null && !required.equals(exchange.getRequestHeaders().getFirst("Authorization"))) {
				exchange.getResponseHeaders().add("WWW-Authenticate", "Basic realm=\"mlflow\"");
				send(exchange, 401, "You are not authenticated. Please see the documentation for details.");
			} else if (method.equals("GET") && path.equals("/version")) {
				send(exchange, 200, "3.0.1");
			} else if (path.equals("/api/2.0/mlflow/experiments/search")) {
				json(exchange, Map.of("experiments", experiments.entrySet().stream().limit(1)
						.map(entry -> Map.of("experiment_id", entry.getValue(), "name", entry.getKey())).toList()));
			} else if (path.equals("/api/2.0/mlflow/experiments/get-by-name")) {
				final String name = URLDecoder.decode(exchange.getRequestURI().getRawQuery()
						.substring("experiment_name=".length()), StandardCharsets.UTF_8);
				final String id = experiments.get(name);
				if (id == null) {
					error(exchange, 404, "RESOURCE_DOES_NOT_EXIST", "Could not find experiment " + name);
				} else {
					json(exchange, Map.of("experiment", Map.of("experiment_id", id, "name", name)));
				}
			} else if (path.equals("/api/2.0/mlflow/experiments/create")) {
				final String id = Integer.toString(ids.incrementAndGet());
				experiments.put(MAPPER.readTree(body).path("name").asText(), id);
				json(exchange, Map.of("experiment_id", id));
			} else if (path.equals("/api/2.0/mlflow/runs/create")) {
				final JsonNode request = MAPPER.readTree(body);
				final String experiment = request.path("experiment_id").asText();
				final var run = new Run("run" + ids.incrementAndGet(), experiment, request.path("run_name").asText());
				request.path("tags").forEach(tag -> run.tags.put(tag.path("key").asText(), tag.path("value").asText()));
				runs.put(run.id, run);
				json(exchange, Map.of("run", Map.of("info", Map.of("run_id", run.id, "experiment_id", experiment,
						"artifact_uri", artifactUriScheme + "/" + experiment + "/" + run.id + "/artifacts"))));
			} else if (path.equals("/api/2.0/mlflow/runs/log-batch")) {
				final JsonNode request = MAPPER.readTree(body);
				final Run run = runs.get(request.path("run_id").asText());
				if (request.path("metrics").size() > 1000 || request.path("params").size() > 100
						|| request.path("tags").size() > 100) {
					error(exchange, 400, "INVALID_PARAMETER_VALUE", "batch too large");
					return;
				}
				if (failMetricBatchesWithStatus != 0 && request.path("metrics").size() > 0) {
					error(exchange, failMetricBatchesWithStatus, "INTERNAL_ERROR", "metrics store unavailable");
					return;
				}
				request.path("metrics").forEach(run.metrics::add);
				request.path("params").forEach(p -> run.params.put(p.path("key").asText(), p.path("value").asText()));
				request.path("tags").forEach(t -> run.tags.put(t.path("key").asText(), t.path("value").asText()));
				json(exchange, Map.of());
			} else if (path.equals("/api/2.0/mlflow/runs/update")) {
				final JsonNode request = MAPPER.readTree(body);
				runs.get(request.path("run_id").asText()).status = request.path("status").asText();
				json(exchange, Map.of());
			} else if (method.equals("PUT") && path.startsWith("/api/2.0/mlflow-artifacts/artifacts/")) {
				artifacts.put(path.substring("/api/2.0/mlflow-artifacts/artifacts/".length()), body);
				json(exchange, Map.of());
			} else {
				error(exchange, 404, "ENDPOINT_NOT_FOUND", "No endpoint " + method + " " + path);
			}
		} catch (RuntimeException exception) {
			error(exchange, 500, "INTERNAL_ERROR", exception.toString());
		}
	}

	private static void json(HttpExchange exchange, Object body) throws IOException {
		exchange.getResponseHeaders().add("Content-Type", "application/json");
		send(exchange, 200, MAPPER.writeValueAsString(body));
	}

	private static void error(HttpExchange exchange, int status, String code, String message) throws IOException {
		exchange.getResponseHeaders().add("Content-Type", "application/json");
		send(exchange, status, MAPPER.writeValueAsString(Map.of("error_code", code, "message", message)));
	}

	private static void send(HttpExchange exchange, int status, String text) throws IOException {
		final byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
		exchange.sendResponseHeaders(status, bytes.length);
		exchange.getResponseBody().write(bytes);
		exchange.close();
	}
}
