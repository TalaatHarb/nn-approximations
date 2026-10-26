package net.talaatharb.nn.tracking;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import net.talaatharb.nn.config.ExampleCatalog;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.tracking.MlflowClient.Metric;
import net.talaatharb.nn.tracking.MlflowRunLogger.Outcome;
import net.talaatharb.nn.tracking.MlflowRunLogger.RunRequest;

class MlflowTrackingTest {

	private FakeMlflowServer server;
	private MlflowClient client;

	@BeforeEach
	void start() throws Exception {
		server = new FakeMlflowServer();
		client = new MlflowClient(server.uri(), java.net.http.HttpClient.newHttpClient(), null);
	}

	@AfterEach
	void stop() {
		server.close();
	}

	@Test
	void reusesOrCreatesExperimentsByName() throws Exception {
		assertEquals("3.0.1", client.version());
		final String created = client.experimentId("nn approximations/xor");
		assertEquals(created, client.experimentId("nn approximations/xor"));
		assertEquals(1, server.experiments.size());
		assertEquals(1, server.requests.stream().filter(request -> request.endsWith("experiments/create")).count());
	}

	@Test
	void splitsLargeBatchesWithinServerLimitsAndSkipsNonFiniteValues() throws Exception {
		final var run = client.createRun(client.experimentId("e"), "r", 1, Map.of("mlflow.user", "me"));
		final var metrics = new ArrayList<Metric>();
		IntStream.range(0, 2500).forEach(step -> metrics.add(new Metric("loss", 1.0 / (step + 1), 5, step)));
		metrics.add(new Metric("loss", Double.NaN, 5, 2500));
		final var params = new LinkedHashMap<String, String>();
		IntStream.range(0, 150).forEach(i -> params.put("p" + i, "x".repeat(i == 0 ? 7000 : 1)));
		client.logBatch(run.runId(), metrics, params, Map.of("t", "v"));

		final var stored = server.runs.get(run.runId());
		assertEquals(2500, stored.metrics("loss").size());
		assertEquals(150, stored.params.size());
		assertEquals(MlflowClient.MAX_PARAM_LENGTH, stored.params.get("p0").length());
		assertEquals("me", stored.tags.get("mlflow.user"));
		assertEquals(5, server.requests.stream().filter(request -> request.endsWith("log-batch")).count());
	}

	@Test
	void uploadsArtifactsThroughTheProxyAndReportsUnsupportedStores() throws Exception {
		final var run = client.createRun(client.experimentId("e"), "r", 1, Map.of());
		client.uploadArtifact(run, "model/MLmodel", "x".getBytes(StandardCharsets.UTF_8));
		assertArrayEquals("x".getBytes(StandardCharsets.UTF_8),
				server.artifacts.get(run.experimentId() + "/" + run.runId() + "/artifacts/model/MLmodel"));
		assertEquals(server.uri() + "/#/experiments/" + run.experimentId() + "/runs/" + run.runId(),
				client.runUrl(run));

		final var s3 = new MlflowClient.RunInfo("1", "r", "s3://bucket/1/r/artifacts");
		final var error = assertThrows(java.io.IOException.class, () -> client.uploadArtifact(s3, "a", new byte[0]));
		assertTrue(error.getMessage().contains("s3://bucket"));
	}

	@Test
	void apiErrorsCarryTheServerMessage() {
		final var error = assertThrows(MlflowClient.MlflowException.class,
				() -> client.updateRun("missing", "FINISHED", 1));
		assertEquals(500, error.status());
		assertTrue(error.getMessage().contains("runs/update"));
	}

	@Test
	void loggerRecordsAWholeRunWithArtifacts() throws Exception {
		final var problems = new CopyOnWriteArrayList<String>();
		final var logger = MlflowRunLogger.start(client,
				new RunRequest("nn-approximations", "XOR run", Map.of("training.learning_rate", "0.3"),
						Map.of("mlflow.source.name", "nn-approximations")),
				problems::add);
		for (int epoch = 1; epoch <= 1500; epoch++) {
			logger.logMetric("loss", 1.0 / epoch, epoch);
		}
		final var model = ExampleCatalog.builtIn("xor");
		final var warnings = new ArrayList<String>();
		final var weights = model.createNetwork(1).copyWeights();
		final var artifacts = RunArtifacts.build(model, weights, warnings);
		final var result = logger.finish(new Outcome("FINISHED", Map.of("final_loss", 0.004),
				Map.of("outcome", "Target reached"), artifacts)).get(20, TimeUnit.SECONDS);

		assertEquals(List.of(), problems);
		assertEquals(List.of(), warnings);
		assertEquals(List.of(), result.warnings());
		final var run = server.runs().get(0);
		assertTrue(result.runUrl().endsWith("/runs/" + run.id));
		assertEquals("FINISHED", run.status);
		assertEquals("XOR run", run.name);
		assertEquals("0.3", run.params.get("training.learning_rate"));
		assertEquals("Target reached", run.tags.get("outcome"));
		assertEquals(1500, run.metrics("loss").size());
		assertEquals(1500, run.metrics("loss").stream().mapToLong(metric -> metric.path("step").asLong()).max()
				.orElseThrow());
		assertEquals(0.004, run.metrics("final_loss").get(0).path("value").asDouble());
		final String prefix = run.experimentId + "/" + run.id + "/artifacts/";
		for (String path : List.of("nn-model.json", "model.pmml", "model/model.onnx", "model/MLmodel",
				"model/requirements.txt", "model/python_env.yaml", "model/conda.yaml")) {
			assertTrue(server.artifacts.containsKey(prefix + path), path);
		}
		final var logged = ModelJson.read(new String(server.artifacts.get(prefix + "nn-model.json"),
				StandardCharsets.UTF_8));
		assertArrayEquals(weights, logged.weights());
		final String mlmodel = new String(server.artifacts.get(prefix + "model/MLmodel"), StandardCharsets.UTF_8);
		assertTrue(mlmodel.contains("loader_module: mlflow.onnx"));
		assertTrue(mlmodel.contains("\"shape\": [-1, 2]}, \"name\": \"input\""));
	}

	@Test
	void loggerReportsFailuresWithoutThrowingIntoTraining() throws Exception {
		server.failMetricBatchesWithStatus = 503;
		final var problems = new CopyOnWriteArrayList<String>();
		final var logger = MlflowRunLogger.start(client, new RunRequest("e", "r", Map.of(), Map.of()),
				problems::add);
		logger.logMetric("loss", 1, 1);
		final var result = logger.finish(new Outcome("KILLED", Map.of(), Map.of(), List.of())).get(20,
				TimeUnit.SECONDS);
		assertEquals("KILLED", server.runs().get(0).status);
		assertTrue(problems.stream().anyMatch(problem -> problem.contains("metrics stopped")), problems::toString);
		assertTrue(result.warnings().contains("some loss values were not logged"));
	}

	@Test
	void unreachableServerFailsTheRunButNotTheCaller() throws Exception {
		server.close();
		final var problems = new CopyOnWriteArrayList<String>();
		final var logger = MlflowRunLogger.start(client, new RunRequest("e", "r", Map.of(), Map.of()),
				problems::add);
		logger.logMetric("loss", 1, 1);
		final var finished = logger.finish(new Outcome("FINISHED", Map.of(), Map.of(), List.of()));
		assertThrows(java.util.concurrent.ExecutionException.class, () -> finished.get(20, TimeUnit.SECONDS));
		assertTrue(problems.stream().anyMatch(problem -> problem.startsWith("MLflow run not created")));
	}

	@Test
	void settingsPersistAndValidate(@TempDir Path directory) throws Exception {
		final Path file = directory.resolve("nested/mlflow.json");
		System.setProperty("nn.mlflow.settings", file.toString());
		try {
			final var defaults = MlflowSettings.load();
			assertFalse(defaults.enabled());
			assertTrue(defaults.validate().isEmpty());
			new MlflowSettings(true, "http://localhost:5000/", "  my experiment ").save();
			final var loaded = MlflowSettings.load();
			assertEquals(new MlflowSettings(true, "http://localhost:5000", "my experiment"), loaded);
			assertTrue(Files.readString(file).contains("\"trackingUri\""));
			assertFalse(new MlflowSettings(true, "ftp://x", null).validate().isEmpty());
			assertFalse(new MlflowSettings(true, "not a uri", null).validate().isEmpty());
			Files.writeString(file, "{broken");
			assertEquals(MlflowSettings.defaults(), MlflowSettings.load());
		} finally {
			System.clearProperty("nn.mlflow.settings");
		}
	}
}
