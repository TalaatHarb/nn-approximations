package net.talaatharb.nn.structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;

import lombok.Getter;
import lombok.ToString;
import net.talaatharb.nn.functions.ActivationFunction;
import net.talaatharb.nn.functions.Functions;
import net.talaatharb.nn.visualization.InspectableNetwork;
import net.talaatharb.nn.visualization.NetworkSnapshot;

@ToString
public class FeedForwardNN implements UnaryOperator<float[]>, InspectableNetwork {

	private final int[] shape;

	@Getter
	private final NeuronLayer[] layers;

	@Getter
	private final int depth;

	public FeedForwardNN(int[] shape) {
		this(shape, java.util.Collections.nCopies(shape.length - 1, Functions.linearFunction()));
	}

	public FeedForwardNN(int[] shape, List<ActivationFunction> functions) {
		if (shape.length < 2 || Arrays.stream(shape).anyMatch(size -> size <= 0)) {
			throw new IllegalArgumentException("A network needs positive input and output layer sizes");
		}
		if (functions.size() != shape.length - 1 || functions.stream().anyMatch(java.util.Objects::isNull)) {
			throw new IllegalArgumentException("Supply one activation function per non-input layer");
		}
		this.shape = shape.clone();
		depth = shape.length - 1;
		layers = new NeuronLayer[depth];
		for (int i = 0; i < depth; i++) {
			layers[i] = new NeuronLayer(shape[i], shape[i + 1], functions.get(i));
		}
	}

	@Override
	public float[] apply(float[] input) {
		if (input.length != shape[0]) {
			throw new IllegalArgumentException("Expected " + shape[0] + " inputs, got " + input.length);
		}
		float[] layerResult = input;
		for (final NeuronLayer layer : layers) {
			layerResult = layer.apply(layerResult);
		}
		return layerResult;
	}

	public static final FeedForwardNN defaultWithShape(int[] shape) {
		return new FeedForwardNN(shape);
	}

	public int[] shape() {
		return shape.clone();
	}

	public void randomize() {
		randomize(new Random());
	}

	public void randomize(Random random) {
		for (final NeuronLayer layer : layers) {
			layer.randomize(random);
		}

	}

	public void setWeights(float[][][] networkWeights) {
		if (networkWeights.length != depth) {
			throw new IllegalArgumentException("Weight layer count does not match network depth");
		}
		for (int i = 0; i < depth; i++) {
			layers[i].validateWeights(networkWeights[i]);
		}
		for (int i = 0; i < depth; i++) {
			layers[i].setWeights(networkWeights[i]);
		}
	}

	public void switchLastLayerFunction(ActivationFunction newFunction) {
		layers[depth - 1].switchFunction(newFunction);
	}

	/** Deep copy of [layer][neuron][bias, weights...]. */
	public float[][][] copyWeights() {
		final var result = new float[depth][][];
		for (int l = 0; l < depth; l++) {
			final var weights = layers[l].weights();
			result[l] = new float[weights.length][];
			for (int n = 0; n < weights.length; n++) {
				result[l][n] = weights[n].clone();
			}
		}
		return result;
	}

	@Override
	public NetworkSnapshot inspect(float[] input) {
		apply(input);
		final var states = new ArrayList<NetworkSnapshot.Layer>();
		final var inputUnits = new ArrayList<NetworkSnapshot.Unit>();
		for (float value : input) {
			inputUnits.add(new NetworkSnapshot.Unit(value, value, 0));
		}
		states.add(new NetworkSnapshot.Layer("Input", inputUnits));
		final var connections = new ArrayList<NetworkSnapshot.Connection>();
		for (int l = 0; l < depth; l++) {
			final var layer = layers[l];
			final var units = new ArrayList<NetworkSnapshot.Unit>();
			for (int n = 0; n < layer.outputSize(); n++) {
				final var neuron = layer.getNeurons()[n];
				units.add(new NetworkSnapshot.Unit(layer.getLastOutput()[n],
						layer.getLastOutputBeforeActivation()[n], neuron.bias()));
				for (int i = 0; i < layer.inputSize(); i++) {
					connections.add(new NetworkSnapshot.Connection(l, i, l + 1, n, neuron.weight(i)));
				}
			}
			states.add(new NetworkSnapshot.Layer((l == depth - 1 ? "Output" : "Hidden " + (l + 1))
					+ " / " + layer.getFunction().getClass().getSimpleName(), units));
		}
		return new NetworkSnapshot("Feed-forward", states, connections);
	}
}
