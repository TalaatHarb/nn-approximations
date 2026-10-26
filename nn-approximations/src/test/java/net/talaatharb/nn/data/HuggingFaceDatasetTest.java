package net.talaatharb.nn.data;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.algorithms.TrainingListener;
import net.talaatharb.nn.config.HuggingFaceDataset;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.metrics.MeanSquareError;

class HuggingFaceDatasetTest {

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

	private DatasetCache.Request request(String split, int count) {
		return new DatasetCache.Request(FakeHuggingFaceServer.REPOSITORY, "default", split, "image", "label",
				FakeHuggingFaceServer.CLASSES, count);
	}

	@Test
	void clientReadsDatasetInfoAndImageSize() throws Exception {
		final var client = server.client(null);
		final var info = client.info(FakeHuggingFaceServer.REPOSITORY);
		assertEquals(1, info.size());
		assertEquals("default", info.get(0).config());
		assertEquals(90L, info.get(0).splits().get("train"));
		assertEquals("image", info.get(0).firstImage().orElseThrow().name());
		assertEquals(FakeHuggingFaceServer.CLASSES, info.get(0).firstClassLabel().orElseThrow().classNames());
		assertArrayEquals(new int[] { 8, 8 }, client.firstImageSize(FakeHuggingFaceServer.REPOSITORY, "default",
				"train", "image").orElseThrow());
		final var missing = assertThrows(IOException.class, () -> client.info("test/missing"));
		assertTrue(missing.getMessage().contains("Not found"), missing.getMessage());
	}

	@Test
	void sendsTokenAndExplainsAuthorizationFailures() throws Exception {
		server.requiredToken = "secret";
		final var denied = assertThrows(IOException.class,
				() -> server.client(null).info(FakeHuggingFaceServer.REPOSITORY));
		assertTrue(denied.getMessage().contains("HF_TOKEN"), denied.getMessage());
		assertEquals(1, server.client("secret").info(FakeHuggingFaceServer.REPOSITORY).size());
	}

	@Test
	void fetchExtractsRequestedSamplesAndServesThemOffline() throws Exception {
		final var cache = new DatasetCache(cacheDirectory, server.client(null));
		assertTrue(cache.cached(request("train", 40)).isEmpty());
		final var messages = new ArrayList<String>();
		final var samples = cache.fetch(request("train", 40), (message, fraction) -> messages.add(message));
		assertEquals(40, samples.size());
		assertFalse(messages.isEmpty());
		for (int i = 0; i < samples.size(); i++) {
			assertEquals(i % 3, samples.get(i).label());
			assertTrue(Files.isRegularFile(samples.get(i).image()));
			assertTrue(samples.get(i).image().startsWith(cacheDirectory));
		}
		final int requests = server.requests.size();
		assertEquals(samples, cache.cached(request("train", 40)).orElseThrow());
		assertEquals(samples.subList(0, 10), cache.fetch(request("train", 10), (m, f) -> { }));
		assertEquals(requests, server.requests.size(), "cached samples must not hit the network");

		// More samples than the split holds: everything is extracted and marked complete.
		final var all = cache.fetch(request("train", 500), (m, f) -> { });
		assertEquals(90, all.size());
		assertEquals(90, cache.cached(request("train", 1000)).orElseThrow().size());
		// A different class list is a different source.
		assertTrue(cache.cached(new DatasetCache.Request(FakeHuggingFaceServer.REPOSITORY, "default", "train",
				"image", "label", List.of("a", "b", "c"), 10)).isEmpty());
	}

	@Test
	void imageLoaderScalesPixelsIntoUnitRange() throws Exception {
		final var cache = new DatasetCache(cacheDirectory, server.client(null));
		final var sample = cache.fetch(request("test", 1), (m, f) -> { }).get(0);
		final float[] pixels = ImageLoader.load(sample.image(), 8, 8, true);
		assertEquals(64, pixels.length);
		int lit = 0;
		for (float pixel : pixels) {
			assertTrue(pixel >= 0 && pixel <= 1);
			lit += pixel > 0.5f ? 1 : 0;
		}
		assertEquals(8, lit, "a horizontal stroke lights one row");
		assertEquals(3 * 16, ImageLoader.load(sample.image(), 4, 4, false).length);
	}

	@Test
	void imageModelRoundTripsAndTrainsFromTheCache() throws Exception {
		server.install();
		System.setProperty(DatasetCache.DIRECTORY_PROPERTY, cacheDirectory.toString());
		final var model = ModelJson.read(FakeHuggingFaceServer.SHAPES_JSON).requireValid();
		assertTrue(model.isImageClassification());
		final var dataset = (HuggingFaceDataset) model.dataset();
		assertEquals(64, dataset.inputSize());
		assertEquals(3, dataset.outputSize());
		assertEquals("px_7_7", dataset.inputNames().get(63));
		assertEquals(dataset, ModelJson.read(ModelJson.write(model)).dataset());
		assertFalse(dataset.isDownloaded());
		assertThrows(HuggingFaceDataset.NotDownloadedException.class, dataset::toTrainingData);

		dataset.fetch((m, f) -> { });
		assertTrue(dataset.isDownloaded());
		final var data = dataset.toTrainingData();
		assertEquals(60, data.getInput().length);
		assertEquals(64, data.getInput()[0].length);
		assertArrayEquals(new float[] { 1, 0, 0 }, data.getOutput()[0]);
		assertEquals(30, dataset.heldOutSamples().size());

		// A tiny network separates the three stroke directions on the held-out split.
		final var network = model.createNetwork(1);
		new StochasticGradientDescent(model.training().learningRate(), new Random(1)).train(network, data,
				model.training().targetLoss(), model.training().maxEpochs(), new MeanSquareError(),
				TrainingListener.NONE, () -> false);
		final var test = dataset.toTrainingData(dataset.heldOutSamples());
		int correct = 0;
		for (int i = 0; i < test.getInput().length; i++) {
			correct += argmax(network.apply(test.getInput()[i])) == argmax(test.getOutput()[i]) ? 1 : 0;
		}
		assertTrue(correct >= 24, "held-out accuracy " + correct + "/30");
	}

	private static int argmax(float[] values) {
		int best = 0;
		for (int i = 1; i < values.length; i++) {
			best = values[i] > values[best] ? i : best;
		}
		return best;
	}

	@Test
	void imageTaskAndDatasetMustBeUsedTogether() throws IOException {
		final var mismatch = ModelJson.read(FakeHuggingFaceServer.SHAPES_JSON.replace("imageClassification", "regression")).validate();
		assertTrue(mismatch.stream().anyMatch(error -> error.contains("imageClassification")), mismatch.toString());
		final var noClasses = ModelJson.read(FakeHuggingFaceServer.SHAPES_JSON.replace("[ \"horizontal\", \"vertical\", \"diagonal\" ]",
				"[ ]")).validate();
		assertTrue(noClasses.stream().anyMatch(error -> error.contains("dataset.classes")), noClasses.toString());
	}
}
