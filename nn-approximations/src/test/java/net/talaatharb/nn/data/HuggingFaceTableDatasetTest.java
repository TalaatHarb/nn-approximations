package net.talaatharb.nn.data;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.config.DataScaling;
import net.talaatharb.nn.config.HuggingFaceDataset;
import net.talaatharb.nn.config.HuggingFaceTableDataset;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.export.OnnxExporter;
import net.talaatharb.nn.metrics.MeanSquareError;
import net.talaatharb.nn.structures.TrainingData;

class HuggingFaceTableDatasetTest {

	static final String FLOWERS_JSON = """
			{
			  "name" : "Flowers",
			  "task" : "classification",
			  "network" : { "inputs" : 4, "layers" : [ { "neurons" : 6, "activation" : "tanh" },
			    { "neurons" : 3, "activation" : "sigmoid" } ] },
			  "initialization" : { "scheme" : "xavierUniform", "seed" : 3 },
			  "training" : { "learningRate" : 0.2, "targetLoss" : 0.005, "maxEpochs" : 200, "shuffleSeed" : 1 },
			  "dataset" : {
			    "type" : "huggingFaceTable",
			    "repository" : "test/flowers",
			    "inputColumns" : [ "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" ],
			    "targetColumn" : "Species",
			    "classes" : [ "setosa", "versicolor", "virginica" ],
			    "samples" : 100,
			    "testSamples" : 15,
			    "seed" : 5
			  }
			}
			""";

	static final String HOUSES_JSON = """
			{
			  "name" : "Houses",
			  "network" : { "inputs" : 2, "layers" : [ { "neurons" : 4, "activation" : "tanh" },
			    { "neurons" : 1, "activation" : "linear" } ] },
			  "initialization" : { "scheme" : "xavierUniform", "seed" : 3 },
			  "training" : { "learningRate" : 0.02, "targetLoss" : 0.002, "maxEpochs" : 400, "shuffleSeed" : 1 },
			  "dataset" : {
			    "type" : "huggingFaceTable",
			    "repository" : "test/houses",
			    "inputColumns" : [ "rooms", "age" ],
			    "targetColumn" : "price",
			    "samples" : 1000,
			    "testSplit" : "test",
			    "testSamples" : 20
			  }
			}
			""";

	@TempDir
	Path cacheDirectory;
	private FakeHuggingFaceServer server;

	@BeforeEach
	void start() throws IOException {
		server = new FakeHuggingFaceServer();
	}

	@AfterEach
	void stop() {
		server.close();
		FakeHuggingFaceServer.uninstall();
		System.clearProperty(DatasetCache.DIRECTORY_PROPERTY);
	}

	private void useServer() {
		server.install();
		System.setProperty(DatasetCache.DIRECTORY_PROPERTY, cacheDirectory.toString());
	}

	@Test
	void clientReadsColumnStatistics() throws Exception {
		final var client = server.client(null);
		final var info = client.info(FakeHuggingFaceServer.FLOWERS).get(0);
		assertEquals("float64", info.feature("PetalWidthCm").orElseThrow().dtype());
		assertTrue(info.firstClassLabel().isEmpty(), "Species is a plain string column");
		final var statistics = client.statistics(FakeHuggingFaceServer.FLOWERS, "default", "train");
		assertEquals(6, statistics.size());
		final var species = statistics.get(5);
		assertEquals("Species", species.name());
		assertTrue(species.isCategorical());
		assertFalse(species.isNumeric());
		assertEquals(FakeHuggingFaceServer.SPECIES, List.copyOf(species.frequencies().keySet()), "sorted names");
		assertTrue(statistics.get(0).isNumeric());
		assertEquals(60.0, statistics.get(0).max());
	}

	@Test
	void fetchTableDownloadsEveryShardAndServesItOffline() throws Exception {
		final var cache = new DatasetCache(cacheDirectory, server.client(null));
		final var columns = List.of("rooms", "age", "price");
		assertTrue(cache.cachedTable(FakeHuggingFaceServer.HOUSES, "default", "train", columns).isEmpty());
		cache.fetchTable(FakeHuggingFaceServer.HOUSES, "default", "train", HuggingFaceClient.Progress.NONE);
		final int requests = server.requests.size();
		final var table = cache.cachedTable(FakeHuggingFaceServer.HOUSES, "default", "train", columns)
				.orElseThrow();
		assertEquals(80, table.rows().size(), "both shards are read");
		assertTrue(table.complete());
		assertInstanceOf(Double.class, table.rows().get(0)[0]);
		assertInstanceOf(Long.class, table.rows().get(0)[1]);
		cache.fetchTable(FakeHuggingFaceServer.HOUSES, "default", "train", HuggingFaceClient.Progress.NONE);
		assertEquals(requests, server.requests.size(), "a cached table must not hit the network");
		final var missing = assertThrows(IllegalArgumentException.class,
				() -> cache.cachedTable(FakeHuggingFaceServer.HOUSES, "default", "train", List.of("rooms", "garage")));
		assertTrue(missing.getMessage().contains("garage") && missing.getMessage().contains("price"),
				missing.getMessage());
	}

	@Test
	void classificationRowsAreShuffledHeldOutScaledAndLearnable() throws Exception {
		useServer();
		final var model = ModelJson.read(FLOWERS_JSON).requireValid();
		final var dataset = (HuggingFaceTableDataset) model.dataset();
		assertEquals(FakeHuggingFaceServer.SPECIES, dataset.outputNames());
		assertFalse(dataset.isDownloaded());
		assertThrows(HuggingFaceDataset.NotDownloadedException.class, dataset::prepare);
		dataset.fetch(HuggingFaceClient.Progress.NONE);
		assertTrue(dataset.isDownloaded());

		final var prepared = dataset.prepare();
		assertEquals(60, prepared.rowsRead());
		assertEquals(1, prepared.rowsSkipped(), "the row with a missing feature");
		assertEquals(15, prepared.test().getInput().length);
		assertEquals(44, prepared.train().getInput().length, "the remaining usable rows (fewer than samples)");
		assertSame(prepared, dataset.prepare(), "memoized");

		// The source is sorted by species; the seeded shuffle mixes the classes into both sets.
		final var trainClasses = new HashSet<Integer>();
		for (int i = 0; i < 10; i++) {
			trainClasses.add(argmax(prepared.train().getOutput()[i]));
		}
		assertTrue(trainClasses.size() > 1, "rows must be shuffled");
		final var reseeded = ModelJson.read(FLOWERS_JSON.replace("\"seed\" : 5", "\"seed\" : 6"));
		assertFalse(java.util.Arrays.deepEquals(prepared.train().getInput(),
				((HuggingFaceTableDataset) reseeded.dataset()).prepare().train().getInput()));
		assertTrue(java.util.Arrays.deepEquals(prepared.train().getInput(),
				((HuggingFaceTableDataset) ModelJson.read(FLOWERS_JSON).dataset()).prepare().train().getInput()),
				"same seed, same rows");

		// Standardized on the training rows; one-hot targets are not scaled.
		final float[][] inputs = prepared.train().getInput();
		for (int c = 0; c < 4; c++) {
			double sum = 0;
			double squares = 0;
			for (float[] row : inputs) {
				sum += row[c];
				squares += row[c] * row[c];
			}
			assertEquals(0, sum / inputs.length, 1e-4);
			assertEquals(1, squares / inputs.length, 1e-3);
		}
		assertFalse(prepared.scaling().scalesOutputs());
		for (float[] target : prepared.train().getOutput()) {
			assertEquals(1, target[0] + target[1] + target[2], 0);
		}
		assertEquals(dataset.scaling(), prepared.scaling());

		final var network = model.createNetwork(0);
		new StochasticGradientDescent(0.2f, new java.util.Random(1)).train(network, prepared.train(), 0.005f, 200,
				new MeanSquareError());
		assertTrue(accuracy(network, prepared.test()) >= 0.9, "held-out flowers should be classified");
	}

	@Test
	void regressionUsesTheTestSplitAndExportsWorkOnRawValues() throws Exception {
		useServer();
		final var model = ModelJson.read(HOUSES_JSON).requireValid();
		final var dataset = (HuggingFaceTableDataset) model.dataset();
		dataset.fetch(HuggingFaceClient.Progress.NONE);
		final var prepared = dataset.prepare();
		assertEquals(80, prepared.train().getInput().length);
		assertEquals(20, prepared.test().getInput().length);
		final var scaling = prepared.scaling();
		assertTrue(scaling.scalesOutputs(), "prices are standardized");
		assertTrue(scaling.outputScale()[0] > 50, "prices spread over hundreds");

		final var network = model.createNetwork(0);
		final float loss = new StochasticGradientDescent(0.02f, new java.util.Random(1)).train(network,
				prepared.train(), 0.002f, 400, new MeanSquareError());
		assertTrue(loss < 0.01, "standardized MSE " + loss);
		double testError = 0;
		for (int i = 0; i < 20; i++) {
			final float predicted = scaling.unscaleOutput(network.apply(prepared.test().getInput()[i]))[0];
			final float actual = scaling.unscaleOutput(prepared.test().getOutput()[i])[0];
			testError += Math.abs(predicted - actual);
		}
		assertTrue(testError / 20 < 25, "mean absolute test error in price units: " + testError / 20);

		// ONNX takes raw column values and returns prices.
		final float[][] raw = { { 3.5f, 10 }, { 8f, 40 } };
		final var environment = OrtEnvironment.getEnvironment();
		try (var session = environment.createSession(OnnxExporter.toBytes(model, network.copyWeights()),
						new OrtSession.SessionOptions());
				var tensor = OnnxTensor.createTensor(environment, raw);
				var result = session.run(Map.of(OnnxExporter.INPUT, tensor))) {
			final float[][] outputs = (float[][]) result.get(OnnxExporter.OUTPUT).orElseThrow().getValue();
			for (int i = 0; i < raw.length; i++) {
				final float expected = scaling.unscaleOutput(network.apply(scaling.scaleInput(raw[i])))[0];
				assertEquals(expected, outputs[i][0], 1e-3 * Math.abs(expected));
				assertEquals(50 * raw[i][0] + 2 * raw[i][1] + 100, outputs[i][0], 40);
			}
			assertTrue(session.getMetadata().getCustomMetadata().get("nn.input_scaling").contains("offset"));
		}
	}

	@Test
	void validationTiesTargetsToTasks() throws Exception {
		final var flowers = ModelJson.read(FLOWERS_JSON);
		assertTrue(flowers.validate().isEmpty());
		assertTrue(String.join("\n", ModelJson.read(FLOWERS_JSON.replace("\"classification\"", "\"regression\""))
				.validate()).contains("needs task classification"));
		final var numericClassification = ModelJson.read(HOUSES_JSON.replace("\"name\" : \"Houses\",",
				"\"name\" : \"Houses\", \"task\" : \"classification\","));
		assertTrue(String.join("\n", numericClassification.validate()).contains("targetScaling none"));
		final var errors = String.join("\n", ModelJson.read(HOUSES_JSON.replace("\"price\"", "\"rooms\"")
				.replace("\"testSplit\" : \"test\"", "\"testSplit\" : \"train\"")).validate());
		assertTrue(errors.contains("cannot also be an input"), errors);
		assertTrue(errors.contains("must differ"), errors);
		final var copy = ModelJson.read(ModelJson.write(flowers));
		assertEquals(flowers.dataset(), copy.dataset());
		assertEquals(DataScaling.Mode.STANDARDIZE, ((HuggingFaceTableDataset) copy.dataset()).inputScaling());
		assertEquals(DataScaling.Mode.NONE, ((HuggingFaceTableDataset) copy.dataset()).effectiveTargetScaling());
	}

	@Test
	void classValuesMatchNamesOrIndices() {
		final var classes = List.of("a", "b", "7");
		assertEquals(1, HuggingFaceTableDataset.classIndex("b", classes));
		assertEquals(2, HuggingFaceTableDataset.classIndex(7L, classes), "integral values match names first");
		assertEquals(2, HuggingFaceTableDataset.classIndex(7.0, classes));
		assertEquals(1, HuggingFaceTableDataset.classIndex(1, classes), "ClassLabel indices");
		assertEquals(-1, HuggingFaceTableDataset.classIndex(1.5, classes));
		assertEquals(-1, HuggingFaceTableDataset.classIndex("c", classes));
		assertEquals(-1, HuggingFaceTableDataset.classIndex(null, classes));
	}

	@Test
	void scalingFitsStandardizedAndMinMaxColumns() {
		final float[][] rows = { { 1, 5 }, { 3, 5 }, { 5, 5 } };
		final float[][] standard = DataScaling.fit(rows, 2, DataScaling.Mode.STANDARDIZE);
		assertArrayEquals(new float[] { 3, 5 }, standard[0]);
		assertEquals(Math.sqrt(8 / 3.0), standard[1][0], 1e-6);
		assertEquals(1, standard[1][1], "constant columns keep scale 1");
		final float[][] minMax = DataScaling.fit(rows, 2, DataScaling.Mode.MIN_MAX);
		assertArrayEquals(new float[] { 1, 5 }, minMax[0]);
		assertArrayEquals(new float[] { 4, 1 }, minMax[1]);
		final var scaling = new DataScaling(minMax[0], minMax[1], new float[] { 10 }, new float[] { 2 });
		assertArrayEquals(new float[] { 0.5f, 0 }, scaling.scaleInput(new float[] { 3, 5 }));
		assertArrayEquals(new float[] { 3, 5 }, scaling.unscaleInput(new float[] { 0.5f, 0 }));
		assertArrayEquals(new float[] { 14 }, scaling.unscaleOutput(new float[] { 2 }));
		assertTrue(DataScaling.identity(2, 1).isIdentity());
		assertThrows(IllegalArgumentException.class,
				() -> new DataScaling(new float[1], new float[] { 0 }, new float[0], new float[0]));
	}

	private static int argmax(float[] values) {
		int best = 0;
		for (int i = 1; i < values.length; i++) {
			best = values[i] > values[best] ? i : best;
		}
		return best;
	}

	private static double accuracy(net.talaatharb.nn.structures.FeedForwardNN network, TrainingData data) {
		int correct = 0;
		for (int i = 0; i < data.getInput().length; i++) {
			correct += argmax(network.apply(data.getInput()[i])) == argmax(data.getOutput()[i]) ? 1 : 0;
		}
		return (double) correct / data.getInput().length;
	}
}
