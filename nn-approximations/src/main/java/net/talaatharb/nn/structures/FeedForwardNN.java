package net.talaatharb.nn.structures;

import java.util.List;
import java.util.function.UnaryOperator;

import lombok.ToString;
import net.talaatharb.nn.functions.ActivationFunction;

@ToString
public class FeedForwardNN implements UnaryOperator<float[]> {

	private final int[] shape;
	private final NeuronLayer[] layers;
	private final int depth;

	public FeedForwardNN(int[] shape) {
		this.shape = shape;
		depth = shape.length - 1;
		layers = new NeuronLayer[depth];
		for (int i = 0; i < depth; i++) {
			layers[i] = NeuronLayer.defaultWithInputOutputSize(shape[i], shape[i + 1]);
		}
	}

	public FeedForwardNN(int[] shape, List<ActivationFunction> functions) {
		this.shape = shape;
		depth = shape.length - 1;
		layers = new NeuronLayer[depth];
		for (int i = 0; i < depth; i++) {
			layers[i] = new NeuronLayer(shape[i], shape[i + 1], functions.get(i));
		}
	}

	@Override
	public float[] apply(float[] input) {
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
		return shape;
	}

	public void randomize() {
		for (final NeuronLayer layer : layers) {
			layer.randomize();
		}

	}

	public void setWeights(float[][][] networkWeights) {
		for (int i = 0; i < depth; i++) {
			layers[i].setWeights(networkWeights[i]);
		}
	}

}
