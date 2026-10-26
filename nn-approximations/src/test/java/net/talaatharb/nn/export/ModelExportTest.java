package net.talaatharb.nn.export;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.xml.parsers.DocumentBuilderFactory;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.w3c.dom.Element;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import net.talaatharb.nn.config.ActivationSpec;
import net.talaatharb.nn.config.ExampleCatalog;
import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.config.NetworkSpec;
import net.talaatharb.nn.config.NetworkSpec.LayerSpec;
import net.talaatharb.nn.structures.FeedForwardNN;

class ModelExportTest {

	private static OrtEnvironment environment;

	@BeforeAll
	static void startRuntime() {
		environment = OrtEnvironment.getEnvironment();
	}

	@AfterAll
	static void closeRuntime() {
		environment.close();
	}

	/** Built-ins plus a model per activation (with parameters off their defaults and a linear hidden layer). */
	static Map<String, ModelDefinition> models() {
		final var models = new LinkedHashMap<String, ModelDefinition>();
		ExampleCatalog.BUILT_IN_IDS.forEach(id -> models.put(id, ExampleCatalog.builtIn(id)));
		final var base = ExampleCatalog.builtIn("xor");
		final List<ActivationSpec> activations = List.of(new ActivationSpec("leakyRelu", Map.of()),
				new ActivationSpec("parametricRelu", Map.of("alpha", 0.3f)), new ActivationSpec("elu", Map.of()),
				new ActivationSpec("scaledElu", Map.of("alpha", 0.7f)),
				new ActivationSpec("piecewiseLinear", Map.of("xMin", -0.5f, "xMax", 2f)),
				new ActivationSpec("step", Map.of("threshold", 0.1f)), new ActivationSpec("relu", Map.of()),
				new ActivationSpec("linear", Map.of()));
		for (var activation : activations) {
			final var network = new NetworkSpec(2, List.of(new LayerSpec(5, activation),
					new LayerSpec(3, new ActivationSpec("linear", Map.of())), new LayerSpec(4, activation),
					new LayerSpec(1, activation)));
			models.put(activation.type() + "-net", new ModelDefinition(null, 1, activation.type() + " net", "", null, network,
					base.initialization(), base.training(), base.dataset(), null));
		}
		return models;
	}

	/** Large weights so every activation is exercised in its saturated and linear regions. */
	static FeedForwardNN network(ModelDefinition model) {
		final var network = model.createNetwork(7);
		final var weights = network.copyWeights();
		for (float[][] layer : weights) {
			for (float[] neuron : layer) {
				for (int i = 0; i < neuron.length; i++) {
					neuron[i] *= 3;
				}
			}
		}
		network.setWeights(weights);
		return network;
	}

	static float[][] inputs(int width) {
		final var random = new Random(3);
		final float[][] inputs = new float[65][width];
		for (int i = 1; i < inputs.length; i++) {
			for (int j = 0; j < width; j++) {
				inputs[i][j] = random.nextFloat() * 4 - 2;
			}
		}
		return inputs;
	}

	@Test
	void onnxGraphsReproduceTheJavaNetwork() throws Exception {
		for (var entry : models().entrySet()) {
			final var model = entry.getValue();
			final var network = network(model);
			final byte[] onnx = OnnxExporter.toBytes(model, network.copyWeights());
			final float[][] inputs = inputs(model.network().inputs());
			try (var session = environment.createSession(onnx, new OrtSession.SessionOptions());
					var tensor = OnnxTensor.createTensor(environment, inputs);
					var result = session.run(Map.of(OnnxExporter.INPUT, tensor))) {
				final float[][] outputs = (float[][]) result.get(OnnxExporter.OUTPUT).orElseThrow().getValue();
				for (int i = 0; i < inputs.length; i++) {
					assertClose(network.apply(inputs[i]), toDouble(outputs[i]), entry.getKey() + " sample " + i);
				}
				final var metadata = session.getMetadata().getCustomMetadata();
				assertEquals(ModelJson.write(model), metadata.get("nn.model_json"), entry.getKey());
				assertEquals(String.join(",", model.dataset().inputNames()), metadata.get("nn.input_names"));
				assertEquals("nn-approximations", session.getMetadata().getProducerName());
			}
		}
	}

	@Test
	void pmmlNetworksReproduceTheJavaNetwork() throws Exception {
		int exported = 0;
		for (var entry : models().entrySet()) {
			final var model = entry.getValue();
			if (PmmlExporter.unsupportedReason(model).isPresent()) {
				continue;
			}
			exported++;
			final var network = network(model);
			final var pmml = new Pmml(PmmlExporter.toXml(model, network.copyWeights()));
			assertEquals(model.network().layers().size(), pmml.layers.size());
			for (float[] input : inputs(model.network().inputs())) {
				assertClose(network.apply(input), pmml.evaluate(input), entry.getKey());
			}
		}
		assertEquals(11, exported, "built-ins plus relu, step and linear networks");
	}

	@Test
	void pmmlExplainsWhatItCannotRepresent() {
		final var models = models();
		assertTrue(PmmlExporter.unsupportedReason(models.get("elu-net")).orElseThrow().contains("ONNX"));
		final var xor = ExampleCatalog.builtIn("xor");
		final var clash = ModelJson.write(xor).replace("\"xor\"", "\"a\"");
		assertTrue(assertDoesNotThrow(() -> PmmlExporter.unsupportedReason(ModelJson.read(clash))).orElseThrow()
				.contains("'a' repeats"));
		assertThrows(IllegalArgumentException.class, () -> PmmlExporter.toXml(models.get("elu-net"),
				network(models.get("elu-net")).copyWeights()));
	}

	@Test
	void exportsRejectWeightsThatDoNotFitOrAreNotFinite(@TempDir Path directory) throws Exception {
		final var model = ExampleCatalog.builtIn("xor");
		final var weights = model.createNetwork(1).copyWeights();
		assertThrows(IllegalArgumentException.class,
				() -> OnnxExporter.toBytes(model, ExampleCatalog.builtIn("sine").createNetwork(1).copyWeights()));
		weights[0][0][0] = Float.NaN;
		assertThrows(IllegalArgumentException.class, () -> OnnxExporter.toBytes(model, weights));
		weights[0][0][0] = 0;
		final Path file = directory.resolve("xor.onnx");
		OnnxExporter.write(model, weights, file);
		PmmlExporter.write(model, weights, directory.resolve("xor.pmml"));
		assertArrayEquals(OnnxExporter.toBytes(model, weights), Files.readAllBytes(file));
		assertTrue(Files.readString(directory.resolve("xor.pmml")).contains("<NeuralNetwork"));
		try (var stream = Files.list(directory)) {
			assertEquals(2, stream.count(), "no temporary files left behind");
		}
	}

	/** Float32 arithmetic in a different order: allow a few ulps relative to the magnitude. */
	private static void assertClose(float[] expected, double[] actual, String message) {
		assertEquals(expected.length, actual.length, message);
		for (int i = 0; i < expected.length; i++) {
			final double tolerance = 2e-5 * Math.max(1, Math.abs(expected[i]));
			assertEquals(expected[i], actual[i], tolerance, message + " output " + i);
		}
	}

	private static double[] toDouble(float[] values) {
		final double[] result = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			result[i] = values[i];
		}
		return result;
	}

	/** Independent reading of the exported document, following the PMML NeuralNetwork semantics. */
	private static final class Pmml {
		private record Neuron(String id, double bias, Map<String, Double> weights) {
		}

		private record Layer(String activation, double threshold, List<Neuron> neurons) {
		}

		final List<String> inputIds = new ArrayList<>();
		final List<Layer> layers = new ArrayList<>();
		final List<String> outputIds = new ArrayList<>();

		Pmml(String xml) throws Exception {
			final var factory = DocumentBuilderFactory.newInstance();
			factory.setNamespaceAware(true);
			final var document = factory.newDocumentBuilder().parse(new ByteArrayInputStream(xml.getBytes("UTF-8")));
			assertEquals("http://www.dmg.org/PMML-4_4", document.getDocumentElement().getNamespaceURI());
			final var inputs = document.getElementsByTagName("NeuralInput");
			for (int i = 0; i < inputs.getLength(); i++) {
				inputIds.add(((Element) inputs.item(i)).getAttribute("id"));
			}
			final var layerNodes = document.getElementsByTagName("NeuralLayer");
			for (int l = 0; l < layerNodes.getLength(); l++) {
				final var layer = (Element) layerNodes.item(l);
				final var neurons = new ArrayList<Neuron>();
				final var neuronNodes = layer.getElementsByTagName("Neuron");
				for (int n = 0; n < neuronNodes.getLength(); n++) {
					final var neuron = (Element) neuronNodes.item(n);
					final var weights = new LinkedHashMap<String, Double>();
					final var connections = neuron.getElementsByTagName("Con");
					for (int c = 0; c < connections.getLength(); c++) {
						final var connection = (Element) connections.item(c);
						weights.put(connection.getAttribute("from"),
								Double.parseDouble(connection.getAttribute("weight")));
					}
					neurons.add(new Neuron(neuron.getAttribute("id"), Double.parseDouble(neuron.getAttribute("bias")),
							weights));
				}
				final String threshold = layer.getAttribute("threshold");
				layers.add(new Layer(layer.getAttribute("activationFunction"),
						threshold.isEmpty() ? 0 : Double.parseDouble(threshold), neurons));
			}
			final var outputs = document.getElementsByTagName("NeuralOutput");
			for (int o = 0; o < outputs.getLength(); o++) {
				outputIds.add(((Element) outputs.item(o)).getAttribute("outputNeuron"));
			}
		}

		double[] evaluate(float[] input) {
			final var values = new LinkedHashMap<String, Double>();
			for (int i = 0; i < input.length; i++) {
				values.put(inputIds.get(i), (double) input[i]);
			}
			for (var layer : layers) {
				final var next = new LinkedHashMap<String, Double>();
				for (var neuron : layer.neurons()) {
					double z = neuron.bias();
					for (var weight : neuron.weights().entrySet()) {
						z += weight.getValue() * values.get(weight.getKey());
					}
					next.put(neuron.id(), switch (layer.activation()) {
						case "identity" -> z;
						case "logistic" -> 1 / (1 + Math.exp(-z));
						case "tanh" -> Math.tanh(z);
						case "rectifier" -> Math.max(0, z);
						case "threshold" -> z > layer.threshold() ? 1.0 : 0.0;
						default -> throw new AssertionError(layer.activation());
					});
				}
				values.putAll(next);
			}
			return outputIds.stream().mapToDouble(values::get).toArray();
		}
	}
}
