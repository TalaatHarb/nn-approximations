package net.talaatharb.nn.config;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.FieldSource;

import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.config.InitializationSpec.Scheme;
import net.talaatharb.nn.metrics.MeanSquareError;
import net.talaatharb.nn.structures.FeedForwardNN;

class ModelDefinitionTest {

	static final List<String> BUILT_INS = ExampleCatalog.BUILT_IN_IDS;
	/** Built-ins whose data is generated offline; Hugging Face datasets are downloaded and tested separately. */
	static final List<String> OFFLINE_BUILT_INS = BUILT_INS.stream()
			.filter(id -> !(ExampleCatalog.builtIn(id).dataset() instanceof HuggingFaceDataset
					|| ExampleCatalog.builtIn(id).dataset() instanceof HuggingFaceTableDataset)).toList();

	@ParameterizedTest
	@FieldSource("BUILT_INS")
	void builtInsRoundTripThroughJson(String id) throws Exception {
		final var model = ExampleCatalog.builtIn(id);
		final var copy = ModelJson.read(ModelJson.write(model));
		assertEquals(ModelJson.write(model), ModelJson.write(copy));
		assertEquals(List.of(), copy.validate());
		if (OFFLINE_BUILT_INS.contains(id)) {
			assertArrayEquals(model.dataset().toTrainingData().getOutput(),
					copy.dataset().toTrainingData().getOutput());
		} else {
			assertEquals(model.dataset(), copy.dataset());
		}
	}

	@ParameterizedTest
	@FieldSource("OFFLINE_BUILT_INS")
	void everyBuiltInLearnsItsFunction(String id) {
		final var model = ExampleCatalog.builtIn(id);
		final var network = model.createNetwork(0);
		final var data = model.dataset().toTrainingData();
		final var training = model.training();
		final var loss = new MeanSquareError();
		final float result = new StochasticGradientDescent(training.learningRate(), training.shuffleRandom())
				.train(network, data, training.targetLoss(), training.maxEpochs(), loss);
		assertTrue(result <= training.targetLoss(), id + " failed to converge: " + result);
		for (int i = 0; i < data.getInput().length; i++) {
			final float[] prediction = network.apply(data.getInput()[i]);
			if (model.isClassification()) {
				assertEquals((int) data.getOutput()[i][0], prediction[0] >= 0.5f ? 1 : 0);
			}
		}
		if (model.dataset() instanceof FunctionDataset function) {
			double unseen = 0;
			for (int i = 0; i < 40; i++) {
				final float[] x = { -0.975f + 0.05f * i };
				final float error = network.apply(x)[0] - function.evaluate(x)[0];
				unseen += error * error;
			}
			assertTrue(unseen / 40 < training.targetLoss() * 1.5, "Poor interpolation for " + id);
		}
	}

	@Test
	void explicitDefaultActivationParametersEqualOmittedOnes() {
		final var explicit = new ActivationSpec("parametricRelu",
				java.util.Map.of("alpha", ActivationType.PARAMETRIC_RELU.defaultValue("alpha")));
		final var omitted = ActivationSpec.of(ActivationType.PARAMETRIC_RELU);
		assertEquals(omitted, explicit);
		assertEquals(omitted.hashCode(), explicit.hashCode());
		assertNotEquals(omitted, new ActivationSpec("parametricRelu", java.util.Map.of("alpha", 0.37f)));
	}

	@Test
	void uniformInitializationMatchesLegacyRandomize() {
		final var model = ExampleCatalog.builtIn("sine");
		final var legacy = new FeedForwardNN(model.network().shape(), model.network().functions());
		legacy.randomize(new Random(42));
		assertArrayEquals(legacy.copyWeights(), model.createNetwork(7).copyWeights());
	}

	@Test
	void initializationIsReproducibleAndScaled() {
		final var base = ExampleCatalog.builtIn("square");
		for (Scheme scheme : Scheme.values()) {
			final var model = base.withInitialization(new InitializationSpec(scheme, -0.5f, 0.5f, null));
			assertArrayEquals(model.createNetwork(5).copyWeights(), model.createNetwork(5).copyWeights());
			assertFalse(java.util.Arrays.deepEquals(model.createNetwork(5).copyWeights(),
					model.createNetwork(6).copyWeights()));
			final var weights = model.createNetwork(5).copyWeights();
			final float limit = switch (scheme) {
				case UNIFORM -> 0.5f;
				case XAVIER_UNIFORM -> (float) Math.sqrt(6.0 / 9);
				case HE_UNIFORM -> (float) Math.sqrt(6.0);
			};
			for (float[][] layer : weights) {
				for (float[] row : layer) {
					if (scheme != Scheme.UNIFORM) {
						assertEquals(0, row[0]);
					}
					for (float w : row) {
						assertTrue(Math.abs(w) <= limit, scheme + " weight out of range: " + w);
					}
				}
			}
		}
	}

	@Test
	void storedWeightsAreRestored() throws Exception {
		final var model = ExampleCatalog.builtIn("xor");
		final var trained = model.createNetwork(1);
		new StochasticGradientDescent(0.3f, new Random(1)).train(trained, model.dataset().toTrainingData(), 0.01f,
				100, new MeanSquareError());
		final var saved = ModelJson.read(ModelJson.write(model.withWeights(trained.copyWeights())));
		assertArrayEquals(trained.copyWeights(), saved.createNetwork(99).copyWeights());
	}

	@Test
	void parameterizedActivationsSerializeAsObjects() throws Exception {
		final String json = ModelJson.write(ExampleCatalog.builtIn("xor")).replace("\"tanh\"",
				"{ \"type\" : \"parametricRelu\", \"alpha\" : 0.25 }");
		final var model = ModelJson.read(json);
		final var activation = model.network().layers().get(0).activation();
		assertEquals(ActivationType.PARAMETRIC_RELU, activation.resolve());
		assertEquals(0.25f, activation.params().get("alpha"));
		assertTrue(ModelJson.write(model).contains("\"alpha\" : 0.25"));
		assertEquals(List.of(), model.validate());
	}

	@Test
	void validationReportsEveryProblem() throws Exception {
		final String json = """
				{
				  "modelType": "rbm", "name": " ",
				  "network": {"inputs": 2, "layers": [{"neurons": 0, "activation": "swish"}, {"neurons": 2, "activation": "linear"}]},
				  "training": {"learningRate": -1, "targetLoss": 0.1, "maxEpochs": 0},
				  "dataset": {"type": "function", "inputs": [{"name": "x", "min": 1, "max": 0, "samples": 3}],
				              "outputs": ["log(x - 5)"]}
				}
				""";
		final var errors = String.join("\n", ModelJson.read(json).validate());
		for (String expected : new String[] { "modelType 'rbm'", "name is required", "neurons must be",
				"unknown activation 'swish'", "learningRate", "maxEpochs", "min must be smaller", "network.inputs is 2" }) {
			assertTrue(errors.contains(expected), "missing '" + expected + "' in:\n" + errors);
		}
		assertThrows(ModelDefinition.InvalidModelException.class, () -> ModelJson.read(json).requireValid());
	}

	@Test
	void oversizedNetworksAreRejected() throws Exception {
		final String json = ModelJson.write(ExampleCatalog.builtIn("mnist")).replace("\"neurons\" : 32",
				"\"neurons\" : 400");
		assertTrue(ModelJson.read(json).validate().toString().contains("at most 250,000 can be visualized"));
	}

	@Test
	void nonFiniteTargetsAndBadExpressionsAreRejected() {
		final var network = new NetworkSpec(1, List.of(new NetworkSpec.LayerSpec(1, ActivationSpec.of(ActivationType.LINEAR))));
		final var training = new TrainingSpec(0.1f, 0.01f, 10, null);
		final var variable = new FunctionDataset.Variable("x", -1, 1, 3);
		assertTrue(new ModelDefinition(null, 0, "n", null, null, network, null, training,
				new FunctionDataset(List.of(variable), List.of("1/x")), null).validate().toString()
				.contains("not finite"));
		assertTrue(new ModelDefinition(null, 0, "n", null, null, network, null, training,
				new FunctionDataset(List.of(variable), List.of("x +")), null).validate().toString()
				.contains("dataset:"));
	}

	@Test
	void unknownPropertiesAreRejected() {
		assertThrows(Exception.class, () -> ModelJson.read(ModelJson.write(ExampleCatalog.builtIn("and"))
				.replace("\"task\"", "\"surprise\" : 1, \"task\"")));
	}

	@Test
	void functionGridVariesLastInputFastest() {
		final var dataset = new FunctionDataset(List.of(new FunctionDataset.Variable("a", 0, 1, 2),
				new FunctionDataset.Variable("b", 0, 2, 3)), List.of("a*10 + b"));
		final var data = dataset.toTrainingData();
		assertArrayEquals(new float[] { 0, 2 }, data.getInput()[2]);
		assertArrayEquals(new float[] { 1, 0 }, data.getInput()[3]);
		assertEquals(11, data.getOutput()[4][0]);
	}

	@Test
	void catalogSavesShadowsAndDeletes(@TempDir Path directory) throws Exception {
		Files.writeString(directory.resolve("broken.json"), "{ nope");
		final var catalog = new ExampleCatalog(directory);
		assertEquals(BUILT_INS.size(), catalog.entries().size());
		assertEquals(1, catalog.problems().size());

		final var xor = ExampleCatalog.builtIn("xor");
		catalog.save("xor", xor.withName("My XOR"));
		final String id = catalog.newIdFor("My Sine! v2");
		assertEquals("my-sine-v2", id);
		catalog.save(id, ExampleCatalog.builtIn("sine").withName("My Sine! v2"));

		final var reloaded = new ExampleCatalog(directory);
		assertEquals(ExampleCatalog.Source.MODIFIED_BUILT_IN, reloaded.find("xor").orElseThrow().source());
		assertEquals("My XOR", reloaded.find("xor").orElseThrow().model().name());
		assertEquals(0, reloaded.entries().indexOf(reloaded.find("xor").orElseThrow()));
		assertEquals(ExampleCatalog.Source.USER, reloaded.find(id).orElseThrow().source());
		assertEquals("my-sine-v2-2", reloaded.newIdFor("my sine v2"));

		assertEquals("XOR", reloaded.delete("xor").orElseThrow().model().name());
		assertTrue(reloaded.delete(id).isEmpty());
		assertThrows(IllegalArgumentException.class, () -> reloaded.delete("and"));
		assertEquals(BUILT_INS.size(), new ExampleCatalog(directory).entries().size());
		assertThrows(IllegalArgumentException.class, () -> catalog.fileFor("../evil"));
	}
}
