package net.talaatharb.nn.data;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * Serves the tiny Parquet fixtures as Hugging Face datasets: {@code test/shapes} (config {@code default},
 * splits {@code train} = 90 rows and {@code test} = 30 rows of 8x8 PNGs, classes horizontal/vertical/diagonal),
 * the iris-like table {@code test/flowers} (split {@code train}, 60 rows sorted by Species, one with a missing
 * PetalLengthCm) and the regression table {@code test/houses} ({@code train} = 80 rows in 2 shards,
 * {@code test} = 20 rows; price = 50 rooms + 2 age + 100 + noise).
 */
public final class FakeHuggingFaceServer implements AutoCloseable {

	public static final String REPOSITORY = "test/shapes";
	public static final List<String> CLASSES = List.of("horizontal", "vertical", "diagonal");
	public static final String FLOWERS = "test/flowers";
	public static final List<String> FLOWER_FEATURES = List.of("SepalLengthCm", "SepalWidthCm", "PetalLengthCm",
			"PetalWidthCm");
	public static final List<String> SPECIES = List.of("setosa", "versicolor", "virginica");
	public static final String HOUSES = "test/houses";

	/** Parquet shard resources per {@code repository/split}. */
	private static final Map<String, List<String>> TABLE_FILES = Map.of(
			"flowers/train", List.of("flowers-train.parquet"),
			"houses/train", List.of("houses-train-0.parquet", "houses-train-1.parquet"),
			"houses/test", List.of("houses-test.parquet"));

	private final HttpServer server;
	public final List<String> requests = Collections.synchronizedList(new ArrayList<>());
	/** When set, requests must carry this bearer token. */
	public volatile String requiredToken;

	public FakeHuggingFaceServer() throws IOException {
		server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		server.createContext("/", this::handle);
		server.start();
	}

	public URI uri() {
		return URI.create("http://127.0.0.1:" + server.getAddress().getPort());
	}

	/** Points {@link HuggingFaceClient#fromEnvironment()} at this server. */
	public void install() {
		System.setProperty(HuggingFaceClient.HUB_PROPERTY, uri().toString());
		System.setProperty(HuggingFaceClient.DATASETS_SERVER_PROPERTY, uri().toString());
	}

	public static void uninstall() {
		System.clearProperty(HuggingFaceClient.HUB_PROPERTY);
		System.clearProperty(HuggingFaceClient.DATASETS_SERVER_PROPERTY);
	}

	public HuggingFaceClient client(String token) {
		return new HuggingFaceClient(uri(), uri(), token);
	}

	private void handle(HttpExchange exchange) throws IOException {
		final String path = exchange.getRequestURI().getPath();
		final String query = exchange.getRequestURI().getRawQuery();
		requests.add(path + (query == null ? "" : "?" + query));
		try (exchange) {
			if (requiredToken != null && !("Bearer " + requiredToken)
					.equals(exchange.getRequestHeaders().getFirst("Authorization"))) {
				send(exchange, 401, "{\"error\":\"Unauthorized\"}");
				return;
			}
			final String dataset = "dataset=test%2Fshapes";
			if (path.equals("/info") && dataset.equals(query)) {
				send(exchange, 200, """
						{"dataset_info": {"default": {"config_name": "default",
						  "features": {"image": {"_type": "Image"},
						    "label": {"names": ["horizontal", "vertical", "diagonal"], "_type": "ClassLabel"}},
						  "splits": {"train": {"name": "train", "num_examples": 90},
						    "test": {"name": "test", "num_examples": 30}}}}}""");
			} else if (path.equals("/first-rows") && query != null && query.startsWith(dataset)) {
				send(exchange, 200, "{\"rows\": [{\"row_idx\": 0, \"row\": {\"image\": {\"src\": \"x\", "
						+ "\"height\": 8, \"width\": 8}, \"label\": 0}}]}");
			} else if (path.matches("/api/datasets/test/shapes/parquet/default/(train|test)")) {
				final String split = path.substring(path.lastIndexOf('/') + 1);
				send(exchange, 200, "[\"" + uri() + "/redirect/" + split + "\"]");
			} else if (path.matches("/redirect/(train|test)")) {
				exchange.getResponseHeaders().add("Location", "/files/" + path.substring(10) + "/0.parquet");
				exchange.sendResponseHeaders(302, -1);
			} else if (path.equals("/info") && "dataset=test%2Fflowers".equals(query)) {
				send(exchange, 200, """
						{"dataset_info": {"default": {"config_name": "default",
						  "features": {"Id": {"dtype": "int64", "_type": "Value"},
						    "SepalLengthCm": {"dtype": "float64", "_type": "Value"},
						    "SepalWidthCm": {"dtype": "float64", "_type": "Value"},
						    "PetalLengthCm": {"dtype": "float64", "_type": "Value"},
						    "PetalWidthCm": {"dtype": "float64", "_type": "Value"},
						    "Species": {"dtype": "string", "_type": "Value"}},
						  "splits": {"train": {"name": "train", "num_examples": 60}}}}}""");
			} else if (path.equals("/statistics") && "dataset=test%2Fflowers&config=default&split=train".equals(query)) {
				final var columns = new ArrayList<String>();
				columns.add(numeric("Id", "int", 1, 60));
				FLOWER_FEATURES.forEach(name -> columns.add(numeric(name, "float", 0.1, 7)));
				columns.add("""
						{"column_name": "Species", "column_type": "string_label", "column_statistics": {"nan_count": 0,
						  "n_unique": 3, "frequencies": {"virginica": 20, "setosa": 20, "versicolor": 20}}}""");
				send(exchange, 200, "{\"num_examples\": 60, \"statistics\": [" + String.join(",", columns)
						+ "], \"partial\": false}");
			} else if (path.equals("/info") && "dataset=test%2Fhouses".equals(query)) {
				send(exchange, 200, """
						{"dataset_info": {"default": {"config_name": "default",
						  "features": {"rooms": {"dtype": "float64", "_type": "Value"},
						    "age": {"dtype": "int64", "_type": "Value"},
						    "price": {"dtype": "float64", "_type": "Value"}},
						  "splits": {"train": {"name": "train", "num_examples": 80},
						    "test": {"name": "test", "num_examples": 20}}}}}""");
			} else if (path.equals("/statistics") && query != null
					&& query.matches("dataset=test%2Fhouses&config=default&split=(train|test)")) {
				send(exchange, 200, "{\"statistics\": [" + numeric("rooms", "float", 2, 9) + ","
						+ numeric("age", "int", 1, 59) + "," + numeric("price", "float", 200, 670) + "]}");
			} else if (path.matches("/api/datasets/test/(flowers|houses)/parquet/default/(train|test)")
					&& TABLE_FILES.containsKey(tableKey(path))) {
				final var files = new ArrayList<String>();
				for (int s = 0; s < TABLE_FILES.get(tableKey(path)).size(); s++) {
					files.add("\"" + uri() + "/tables/" + tableKey(path) + "/" + s + ".parquet\"");
				}
				send(exchange, 200, "[" + String.join(",", files) + "]");
			} else if (path.matches("/tables/(flowers|houses)/(train|test)/\\d\\.parquet")) {
				final String key = path.substring(8, path.lastIndexOf('/'));
				final int shard = path.charAt(path.lastIndexOf('/') + 1) - '0';
				final var shards = TABLE_FILES.getOrDefault(key, List.of());
				if (shard >= shards.size()) {
					send(exchange, 404, "{\"error\":\"Not found\"}");
					return;
				}
				try (InputStream input = FakeHuggingFaceServer.class.getResourceAsStream("/data/" + shards.get(shard))) {
					final byte[] bytes = input.readAllBytes();
					exchange.sendResponseHeaders(200, bytes.length);
					exchange.getResponseBody().write(bytes);
				}
			} else if (path.matches("/files/(train|test)/0\\.parquet")) {
				final String split = path.substring(7, path.indexOf("/0.parquet"));
				try (InputStream input = FakeHuggingFaceServer.class
						.getResourceAsStream("/data/tiny-" + split + ".parquet")) {
					final byte[] bytes = input.readAllBytes();
					exchange.sendResponseHeaders(200, bytes.length);
					exchange.getResponseBody().write(bytes);
				}
			} else {
				send(exchange, 404, "{\"error\":\"Not found\"}");
			}
		}
	}

	private static String tableKey(String apiPath) {
		final String[] parts = apiPath.split("/");
		return parts[4] + "/" + parts[parts.length - 1];
	}

	private static String numeric(String name, String type, double min, double max) {
		return "{\"column_name\": \"" + name + "\", \"column_type\": \"" + type
				+ "\", \"column_statistics\": {\"nan_count\": 0, \"min\": " + min + ", \"max\": " + max
				+ ", \"mean\": " + (min + max) / 2 + ", \"std\": 1.0}}";
	}

	private static void send(HttpExchange exchange, int status, String body) throws IOException {
		final byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
		exchange.getResponseHeaders().add("Content-Type", "application/json");
		exchange.sendResponseHeaders(status, bytes.length);
		exchange.getResponseBody().write(bytes);
	}

	@Override
	public void close() {
		server.stop(0);
	}

	/** An 8x8 grayscale model over this server's dataset. */
	public static final String SHAPES_JSON = """
			{
			  "modelType" : "feedForward",
			  "formatVersion" : 1,
			  "name" : "Shapes",
			  "task" : "imageClassification",
			  "network" : {
			    "inputs" : 64,
			    "layers" : [ { "neurons" : 6, "activation" : "sigmoid" }, { "neurons" : 3, "activation" : "sigmoid" } ]
			  },
			  "initialization" : { "scheme" : "xavierUniform", "seed" : 3 },
			  "training" : { "learningRate" : 0.5, "targetLoss" : 0.001, "maxEpochs" : 60, "shuffleSeed" : 1 },
			  "dataset" : {
			    "type" : "huggingFace",
			    "repository" : "test/shapes",
			    "classes" : [ "horizontal", "vertical", "diagonal" ],
			    "samples" : 60,
			    "testSplit" : "test",
			    "testSamples" : 30,
			    "image" : { "width" : 8, "height" : 8, "grayscale" : true }
			  }
			}
			""";
}
