package net.talaatharb.nn.export;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import net.talaatharb.nn.config.ActivationSpec;
import net.talaatharb.nn.config.ActivationType;
import net.talaatharb.nn.config.DataScaling;
import net.talaatharb.nn.config.DatasetSpec;
import net.talaatharb.nn.config.HuggingFaceTableDataset;
import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;

/**
 * Writes a trained feed-forward model as an ONNX graph (IR 8, default opset 13), loadable by onnxruntime, Netron,
 * PyTorch/TensorFlow converters, ML.NET and similar tools.
 * <p>
 * The graph has one input {@code input} of shape [batch, inputs] and one output {@code output} of shape
 * [batch, outputs]. Each layer is a {@code Gemm} followed by its activation; every activation of this project maps to
 * standard operators, so the exported graph reproduces {@link net.talaatharb.nn.structures.FeedForwardNN#apply}.
 */
public final class OnnxExporter {

	public static final String INPUT = "input";
	public static final String OUTPUT = "output";
	public static final int OPSET = 13;
	static final int IR_VERSION = 8;

	private static final int FLOAT = 1;
	private static final int ATTRIBUTE_FLOAT = 1;
	private static final int ATTRIBUTE_INT = 2;

	private OnnxExporter() {
	}

	public static void write(ModelDefinition model, float[][][] weights, Path path) throws IOException {
		write(model, weights, model.dataset().scaling(), path);
	}

	public static void write(ModelDefinition model, float[][][] weights, DataScaling scaling, Path path)
			throws IOException {
		ExportFiles.writeAtomically(path, toBytes(model, weights, scaling));
	}

	/** Uses the dataset's {@link DatasetSpec#scaling() scaling}, so consumers pass raw column values. */
	public static byte[] toBytes(ModelDefinition model, float[][][] weights) {
		return toBytes(model, weights, model.dataset().scaling());
	}

	/**
	 * A non-identity scaling becomes {@code Sub}/{@code Div} nodes before the first layer and {@code Mul}/{@code Add}
	 * after the last one, so the graph takes raw column values and returns predictions in target units.
	 */
	public static byte[] toBytes(ModelDefinition model, float[][][] weights, DataScaling scaling) {
		ExportFiles.requireCompatible(model, weights);
		ExportFiles.requireCompatible(model, scaling);
		final var nodes = new ArrayList<byte[]>();
		final var initializers = new ArrayList<byte[]>();
		final var layers = model.network().layers();
		String previous = INPUT;
		if (scaling.scalesInputs()) {
			final long[] dims = { scaling.inputOffset().length };
			initializers.add(tensor("input_offset", dims, scaling.inputOffset()));
			initializers.add(tensor("input_scale", dims, scaling.inputScale()));
			nodes.add(node("input_center", "Sub", List.of(INPUT, "input_offset"), "input_centered", w -> {
			}));
			nodes.add(node("input_normalize", "Div", List.of("input_centered", "input_scale"), "input_scaled", w -> {
			}));
			previous = "input_scaled";
		}
		final boolean unscale = scaling.scalesOutputs();
		for (int l = 0; l < layers.size(); l++) {
			final boolean last = l == layers.size() - 1;
			final var activation = layers.get(l).activation();
			final boolean linear = activation.resolve() == ActivationType.LINEAR;
			final String a = last ? (unscale ? "network_output" : OUTPUT) : "layer" + l + "_a";
			final String z = linear ? a : "layer" + l + "_z";
			final float[][] layer = weights[l];
			final int neurons = layer.length;
			final int fanIn = layer[0].length - 1;
			final float[] matrix = new float[neurons * fanIn];
			final float[] bias = new float[neurons];
			for (int n = 0; n < neurons; n++) {
				bias[n] = layer[n][0];
				System.arraycopy(layer[n], 1, matrix, n * fanIn, fanIn);
			}
			initializers.add(tensor("layer" + l + "_W", new long[] { neurons, fanIn }, matrix));
			initializers.add(tensor("layer" + l + "_B", new long[] { neurons }, bias));
			nodes.add(node("layer" + l + "_gemm", "Gemm", List.of(previous, "layer" + l + "_W", "layer" + l + "_B"),
					z, w -> w.message(5, attribute -> intAttribute(attribute, "transB", 1))));
			if (!linear) {
				activation(nodes, initializers, "layer" + l, activation, z, a);
			}
			previous = a;
		}
		if (unscale) {
			final long[] dims = { scaling.outputOffset().length };
			initializers.add(tensor("output_scale", dims, scaling.outputScale()));
			initializers.add(tensor("output_offset", dims, scaling.outputOffset()));
			nodes.add(node("output_rescale", "Mul", List.of(previous, "output_scale"), "output_rescaled", w -> {
			}));
			nodes.add(node("output_shift", "Add", List.of("output_rescaled", "output_offset"), OUTPUT, w -> {
			}));
		}

		final var definition = model.withWeights(null);
		final var metadata = new LinkedHashMap<String, String>();
		metadata.put("nn.name", model.name());
		metadata.put("nn.task", model.task().name().toLowerCase(java.util.Locale.ROOT));
		metadata.put("nn.input_names", String.join(",", model.dataset().inputNames()));
		metadata.put("nn.output_names", String.join(",", model.dataset().outputNames()));
		if (model.dataset() instanceof net.talaatharb.nn.config.HuggingFaceDataset images) {
			final var image = images.image();
			metadata.put("nn.input_image", image.width() + "x" + image.height() + " "
					+ (image.grayscale() ? "grayscale" : "RGB interleaved per pixel")
					+ ", row-major, pixel values scaled to 0..1");
			metadata.put("nn.dataset", images.repository() + " " + images.config() + "/" + images.split());
		}
		if (model.dataset() instanceof HuggingFaceTableDataset table) {
			metadata.put("nn.dataset", table.repository() + " " + table.config() + "/" + table.split() + " -> "
					+ table.targetColumn());
		}
		if (!scaling.isIdentity()) {
			metadata.put("nn.input_scaling", "raw inputs; the graph computes (input - offset) / scale with offset="
					+ Arrays.toString(scaling.inputOffset()) + " scale=" + Arrays.toString(scaling.inputScale()));
			metadata.put("nn.output_scaling", "outputs in target units: network * scale + offset with scale="
					+ Arrays.toString(scaling.outputScale()) + " offset=" + Arrays.toString(scaling.outputOffset()));
		}
		metadata.put("nn.model_json", ModelJson.write(definition));
		final var graph = new ProtoWriter();
		nodes.forEach(node -> graph.bytes(1, node));
		graph.string(2, safeName(model.name()));
		initializers.forEach(initializer -> graph.bytes(5, initializer));
		graph.string(10, model.description());
		graph.bytes(11, valueInfo(INPUT, model.network().inputs(),
				"Columns: " + String.join(", ", model.dataset().inputNames())));
		graph.bytes(12, valueInfo(OUTPUT, layers.get(layers.size() - 1).neurons(),
				"Columns: " + String.join(", ", model.dataset().outputNames())));

		final var proto = new ProtoWriter().varint(1, IR_VERSION).string(2, "nn-approximations").string(3, "1")
				.string(4, "net.talaatharb.nn").varint(5, 1).string(6, model.description())
				.bytes(7, graph.toByteArray()).message(8, opset -> opset.string(1, "").varint(2, OPSET));
		metadata.forEach((key, value) -> proto.message(14, entry -> entry.string(1, key).string(2, value)));
		return proto.toByteArray();
	}

	private static void activation(List<byte[]> nodes, List<byte[]> initializers, String prefix, ActivationSpec spec,
			String in, String out) {
		final var type = spec.resolve();
		final Map<String, Float> p = spec.params();
		switch (type) {
			case SIGMOID -> nodes.add(node(prefix + "_act", "Sigmoid", List.of(in), out, w -> {
			}));
			case TANH -> nodes.add(node(prefix + "_act", "Tanh", List.of(in), out, w -> {
			}));
			case RELU -> nodes.add(node(prefix + "_act", "Relu", List.of(in), out, w -> {
			}));
			case LEAKY_RELU, PARAMETRIC_RELU -> {
				final float alpha = type == ActivationType.LEAKY_RELU ? 0.01f : value(p, type, "alpha");
				nodes.add(node(prefix + "_act", "LeakyRelu", List.of(in), out,
						w -> w.message(5, a -> floatAttribute(a, "alpha", alpha))));
			}
			case ELU, SCALED_ELU -> {
				final float alpha = type == ActivationType.ELU ? 1.0f : value(p, type, "alpha");
				nodes.add(node(prefix + "_act", "Elu", List.of(in), out,
						w -> w.message(5, a -> floatAttribute(a, "alpha", alpha))));
			}
			case PIECEWISE_LINEAR -> {
				// 0 below xMin, 1 above xMax, linear in between: exactly HardSigmoid(alpha = slope, beta = offset).
				final float min = value(p, type, "xMin");
				final float slope = 1.0f / (value(p, type, "xMax") - min);
				final float offset = -slope * min;
				nodes.add(node(prefix + "_act", "HardSigmoid", List.of(in), out, w -> w
						.message(5, a -> floatAttribute(a, "alpha", slope))
						.message(5, a -> floatAttribute(a, "beta", offset))));
			}
			case STEP -> {
				initializers.add(tensor(prefix + "_threshold", new long[0], new float[] { value(p, type, "threshold") }));
				nodes.add(node(prefix + "_above", "Greater", List.of(in, prefix + "_threshold"), prefix + "_mask",
						w -> {
						}));
				nodes.add(node(prefix + "_act", "Cast", List.of(prefix + "_mask"), out,
						w -> w.message(5, a -> intAttribute(a, "to", FLOAT))));
			}
			case LINEAR -> throw new IllegalStateException("linear layers need no activation node");
		}
	}

	private static float value(Map<String, Float> params, ActivationType type, String name) {
		return params.getOrDefault(name, type.defaultValue(name));
	}

	private static byte[] node(String name, String op, List<String> inputs, String output,
			java.util.function.Consumer<ProtoWriter> attributes) {
		final var node = new ProtoWriter();
		inputs.forEach(input -> node.string(1, input));
		node.string(2, output).string(3, name).string(4, op);
		attributes.accept(node);
		return node.toByteArray();
	}

	private static void floatAttribute(ProtoWriter attribute, String name, float value) {
		attribute.string(1, name).float32(2, value).varint(20, ATTRIBUTE_FLOAT);
	}

	private static void intAttribute(ProtoWriter attribute, String name, long value) {
		attribute.string(1, name).varint(3, value).varint(20, ATTRIBUTE_INT);
	}

	private static byte[] tensor(String name, long[] dims, float[] values) {
		final var tensor = new ProtoWriter();
		for (long dim : dims) {
			tensor.varint(1, dim);
		}
		return tensor.varint(2, FLOAT).string(8, name).bytes(9, ProtoWriter.littleEndian(values)).toByteArray();
	}

	private static byte[] valueInfo(String name, int columns, String doc) {
		return new ProtoWriter().string(1, name)
				.message(2, type -> type.message(1, tensorType -> tensorType.varint(1, FLOAT)
						.message(2, shape -> shape.message(1, dim -> dim.string(2, "batch"))
								.message(1, dim -> dim.varint(1, columns)))))
				.string(3, doc).toByteArray();
	}

	private static String safeName(String name) {
		final String cleaned = name.replaceAll("[^A-Za-z0-9_]+", "_");
		return cleaned.isEmpty() ? "network" : cleaned;
	}
}
