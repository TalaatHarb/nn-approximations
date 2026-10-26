package net.talaatharb.nn.data;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Talks to the public Hugging Face APIs: the datasets server for dataset metadata and the Hub's
 * Parquet export (every public dataset is converted to Parquet) for the data itself. A token in
 * {@code HF_TOKEN} is sent as a bearer token so gated datasets work.
 */
public final class HuggingFaceClient {

	public static final String HUB_PROPERTY = "nn.huggingface.hub";
	public static final String DATASETS_SERVER_PROPERTY = "nn.huggingface.datasetsServer";
	public static final String DEFAULT_HUB = "https://huggingface.co";
	public static final String DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";

	public static final Pattern REPOSITORY = Pattern.compile("[A-Za-z0-9][\\w.-]*(/[A-Za-z0-9][\\w.-]*)?");
	public static final Pattern NAME = Pattern.compile("[\\w][\\w.-]*");

	/** A column description from the dataset card ({@code _type} Image, ClassLabel, Value, ...). */
	public record Feature(String name, String type, String dtype, List<String> classNames) {
		public boolean isImage() {
			return "Image".equals(type);
		}

		public boolean isClassLabel() {
			return "ClassLabel".equals(type);
		}
	}

	public record ConfigInfo(String config, List<Feature> features, Map<String, Long> splits) {
		public Optional<Feature> feature(String name) {
			return features.stream().filter(feature -> feature.name().equals(name)).findFirst();
		}

		public Optional<Feature> firstImage() {
			return features.stream().filter(Feature::isImage).findFirst();
		}

		public Optional<Feature> firstClassLabel() {
			return features.stream().filter(Feature::isClassLabel).findFirst();
		}
	}

	public interface Progress {
		Progress NONE = (message, fraction) -> {
		};

		/** @param fraction 0..1, or negative when unknown */
		void update(String message, double fraction);
	}

	private static final ObjectMapper JSON = new ObjectMapper();

	private final URI hub;
	private final URI datasetsServer;
	private final String token;
	private final HttpClient http;

	public HuggingFaceClient(URI hub, URI datasetsServer, String token) {
		this.hub = trim(hub);
		this.datasetsServer = trim(datasetsServer);
		this.token = token == null || token.isBlank() ? null : token.trim();
		this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(20))
				.followRedirects(HttpClient.Redirect.NORMAL).build();
	}

	/** Endpoints from system properties, then {@code HF_ENDPOINT}, then the public defaults. */
	public static HuggingFaceClient fromEnvironment() {
		final String hub = firstNonBlank(System.getProperty(HUB_PROPERTY), System.getenv("HF_ENDPOINT"), DEFAULT_HUB);
		final String server = firstNonBlank(System.getProperty(DATASETS_SERVER_PROPERTY), DEFAULT_DATASETS_SERVER);
		return new HuggingFaceClient(URI.create(hub), URI.create(server), System.getenv("HF_TOKEN"));
	}

	private static String firstNonBlank(String... values) {
		for (String value : values) {
			if (value != null && !value.isBlank()) {
				return value.trim();
			}
		}
		return null;
	}

	private static URI trim(URI uri) {
		final String text = uri.toString();
		return URI.create(text.endsWith("/") ? text.substring(0, text.length() - 1) : text);
	}

	public URI hub() {
		return hub;
	}

	public static String datasetPage(String repository) {
		return DEFAULT_HUB + "/datasets/" + repository;
	}

	/** Features and split sizes of every configuration of a dataset. */
	public List<ConfigInfo> info(String repository) throws IOException, InterruptedException {
		requireRepository(repository);
		final var root = getJson(URI.create(datasetsServer + "/info?dataset=" + encode(repository)));
		final var configs = new ArrayList<ConfigInfo>();
		root.path("dataset_info").fields().forEachRemaining(entry -> configs.add(config(entry.getKey(),
				entry.getValue())));
		if (configs.isEmpty()) {
			throw new IOException("Hugging Face returned no configurations for " + repository);
		}
		return configs;
	}

	private static ConfigInfo config(String name, JsonNode node) {
		final var features = new ArrayList<Feature>();
		node.path("features").fields().forEachRemaining(entry -> {
			final var feature = entry.getValue();
			final var names = new ArrayList<String>();
			feature.path("names").forEach(item -> names.add(item.asText()));
			features.add(new Feature(entry.getKey(), feature.path("_type").asText(""), feature.path("dtype").asText(""),
					List.copyOf(names)));
		});
		final var splits = new LinkedHashMap<String, Long>();
		node.path("splits").fields().forEachRemaining(entry -> splits.put(entry.getKey(),
				entry.getValue().path("num_examples").asLong(-1)));
		return new ConfigInfo(node.path("config_name").asText(name), List.copyOf(features), splits);
	}

	/**
	 * Per-column statistics of a split computed by the datasets server ({@code column_type} int, float,
	 * string_label, class_label, ...). Categorical columns list their values in {@code frequencies}.
	 */
	public record ColumnStatistics(String name, String type, Double min, Double max, Double mean, Double std,
			Map<String, Long> frequencies) {
		public boolean isNumeric() {
			return "int".equals(type) || "float".equals(type);
		}

		public boolean isCategorical() {
			return !frequencies.isEmpty();
		}
	}

	/** Column statistics of one split; empty when the server has not computed them. */
	public List<ColumnStatistics> statistics(String repository, String config, String split)
			throws IOException, InterruptedException {
		requireRepository(repository);
		final var root = getJson(URI.create(datasetsServer + "/statistics?dataset=" + encode(repository) + "&config="
				+ encode(config) + "&split=" + encode(split)));
		final var columns = new ArrayList<ColumnStatistics>();
		for (var column : root.path("statistics")) {
			final var stats = column.path("column_statistics");
			final var frequencies = new java.util.TreeMap<String, Long>();
			stats.path("frequencies").fields().forEachRemaining(entry -> frequencies.put(entry.getKey(),
					entry.getValue().asLong()));
			columns.add(new ColumnStatistics(column.path("column_name").asText(), column.path("column_type").asText(""),
					number(stats.path("min")), number(stats.path("max")), number(stats.path("mean")),
					number(stats.path("std")), java.util.Collections.unmodifiableMap(frequencies)));
		}
		return List.copyOf(columns);
	}

	private static Double number(JsonNode node) {
		return node.isNumber() ? node.asDouble() : null;
	}

	/** Width and height of the first image in a split, when the datasets server reports it. */
	public Optional<int[]> firstImageSize(String repository, String config, String split, String column)
			throws IOException, InterruptedException {
		requireRepository(repository);
		final var root = getJson(URI.create(datasetsServer + "/first-rows?dataset=" + encode(repository) + "&config="
				+ encode(config) + "&split=" + encode(split)));
		for (var row : root.path("rows")) {
			final var image = row.path("row").path(column);
			if (image.path("width").isInt() && image.path("height").isInt()) {
				return Optional.of(new int[] { image.path("width").asInt(), image.path("height").asInt() });
			}
		}
		return Optional.empty();
	}

	/** URLs of the Parquet shards of one split, in order. */
	public List<URI> parquetFiles(String repository, String config, String split)
			throws IOException, InterruptedException {
		requireRepository(repository);
		requireName("config", config);
		requireName("split", split);
		final var root = getJson(URI.create(hub + "/api/datasets/" + repository + "/parquet/" + encode(config) + "/"
				+ encode(split)));
		final var files = new ArrayList<URI>();
		for (var item : root) {
			files.add(URI.create(item.asText()));
		}
		if (files.isEmpty()) {
			throw new IOException("No Parquet files for " + repository + " " + config + "/" + split);
		}
		return files;
	}

	/** Streams a file to {@code target} (atomically), reporting progress. */
	public void download(URI uri, Path target, Progress progress) throws IOException, InterruptedException {
		final var response = http.send(request(uri).build(), HttpResponse.BodyHandlers.ofInputStream());
		if (response.statusCode() != 200) {
			response.body().close();
			throw new IOException(failure(uri, response.statusCode()));
		}
		final long total = response.headers().firstValueAsLong("Content-Length").orElse(-1);
		Files.createDirectories(target.getParent());
		final Path partial = target.resolveSibling(target.getFileName() + ".part");
		try (InputStream input = response.body(); var output = Files.newOutputStream(partial)) {
			final byte[] buffer = new byte[1 << 16];
			long done = 0;
			long reported = 0;
			for (int read; (read = input.read(buffer)) >= 0;) {
				if (Thread.currentThread().isInterrupted()) {
					throw new InterruptedException("Download cancelled");
				}
				output.write(buffer, 0, read);
				done += read;
				if (done - reported >= 1 << 18) {
					reported = done;
					progress.update(String.format(java.util.Locale.ROOT, "Downloading %s: %.1f MB%s",
							fileName(uri), done / 1e6, total > 0 ? String.format(java.util.Locale.ROOT, " of %.1f MB",
									total / 1e6) : ""), total > 0 ? (double) done / total : -1);
				}
			}
		} catch (IOException | InterruptedException | RuntimeException exception) {
			Files.deleteIfExists(partial);
			throw exception;
		}
		Files.move(partial, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
	}

	private static String fileName(URI uri) {
		final String path = uri.getPath();
		return path == null ? uri.toString() : path.substring(path.lastIndexOf('/') + 1);
	}

	private JsonNode getJson(URI uri) throws IOException, InterruptedException {
		final var response = http.send(request(uri).header("Accept", "application/json").build(),
				HttpResponse.BodyHandlers.ofString());
		if (response.statusCode() != 200) {
			throw new IOException(failure(uri, response.statusCode()) + detail(response.body()));
		}
		try {
			return JSON.readTree(response.body());
		} catch (IOException exception) {
			throw new IOException("Unexpected response from " + uri + ": " + exception.getMessage(), exception);
		}
	}

	private static String detail(String body) {
		try {
			final var error = JSON.readTree(body).path("error").asText("");
			return error.isEmpty() ? "" : " (" + error + ")";
		} catch (IOException | RuntimeException exception) {
			return "";
		}
	}

	private HttpRequest.Builder request(URI uri) {
		final var builder = HttpRequest.newBuilder(uri).timeout(Duration.ofMinutes(5)).header("User-Agent",
				"nn-approximations");
		if (token != null) {
			builder.header("Authorization", "Bearer " + token);
		}
		return builder;
	}

	private String failure(URI uri, int status) {
		return switch (status) {
			case 401, 403 -> "Hugging Face denied access to " + uri + " (HTTP " + status + "); "
					+ (token == null ? "gated datasets need an access token in the HF_TOKEN environment variable"
							: "check that HF_TOKEN may read this dataset");
			case 404 -> "Not found on Hugging Face: " + uri + " (check the repository, config and split names)";
			default -> "Hugging Face request failed with HTTP " + status + ": " + uri;
		};
	}

	private static String encode(String value) {
		return URLEncoder.encode(value, StandardCharsets.UTF_8);
	}

	public static void requireRepository(String repository) {
		if (repository == null || !REPOSITORY.matcher(repository).matches()) {
			throw new IllegalArgumentException("Repository must look like 'owner/name', e.g. ylecun/mnist");
		}
	}

	private static void requireName(String label, String value) {
		if (value == null || !NAME.matcher(value).matches()) {
			throw new IllegalArgumentException(label + " '" + value + "' may only use letters, digits, '_', '.', '-'");
		}
	}
}
