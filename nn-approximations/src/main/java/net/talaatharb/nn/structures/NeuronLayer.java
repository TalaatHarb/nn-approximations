package net.talaatharb.nn.structures;

import java.util.Random;
import java.util.function.UnaryOperator;

public class NeuronLayer implements UnaryOperator<float[]> {

	private static final Random RANDOM = new Random();

	public static final NeuronLayer defaultWithInputOutputSize(int inputSize, int outputSize) {
		return new NeuronLayer(inputSize, outputSize);
	}
	public static NeuronLayer defaultWithWeights(float[][] layerWeights) {
		return new NeuronLayer(layerWeights);
	}
	private final int inputSize;

	private final Neuron[] neurons;

	private final int outputSize;

	public NeuronLayer(float[][] layerWeights) {
		outputSize = layerWeights.length;
		inputSize = layerWeights[0].length;
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = Neuron.defaultWithWeights(layerWeights[i]);
		}
	}

	public NeuronLayer(int inputSize, int outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = Neuron.defaultWithSize(inputSize);
		}
	}

	public NeuronLayer(float[][] layerWeights, UnaryOperator<Float> function) {
		outputSize = layerWeights.length;
		inputSize = layerWeights[0].length;
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = new Neuron(layerWeights[i], function);
		}
	}
	@Override
	public float[] apply(float[] input) {
		final float[] output = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			output[i] = neurons[i].apply(input);
		}
		return output;
	}

	public int inputSize() {
		return inputSize;
	}

	public int outputSize() {
		return neurons.length;
	}

	public void randomize() {
		for (int i = 0; i < outputSize; i++) {
			final Neuron neuron = neurons[i];
			neuron.updateBias(RANDOM.nextFloat(-1.0f, 1.0f));
			for (int j = 0; j < inputSize; j++) {
				neuron.updateWeight(j, RANDOM.nextFloat(-1.0f, 1.0f));
			}
		}

	}

	public void setWeights(float[][] layerWeights) {
		final int min = Math.min(outputSize, layerWeights.length);
		for (int i = 0; i < min; i++) {
			neurons[i].updateWeights(layerWeights[i]);
		}

	}

	public float[][] weights() {
		final float[][] layerWeights = new float[outputSize][];
		for (int i = 0; i < outputSize; i++) {
			layerWeights[i] = neurons[i].weights();
		}
		return layerWeights;
	}

}
