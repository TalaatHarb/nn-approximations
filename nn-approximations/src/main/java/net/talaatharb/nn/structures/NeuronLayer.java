package net.talaatharb.nn.structures;

import java.util.Random;
import java.util.function.UnaryOperator;

import lombok.Getter;
import lombok.ToString;
import net.talaatharb.nn.functions.ActivationFunction;
import net.talaatharb.nn.functions.Functions;

@ToString
public class NeuronLayer implements UnaryOperator<float[]> {

	private static final Random RANDOM = new Random();
	
	@Getter
	private final ActivationFunction function;

	public static final NeuronLayer defaultWithInputOutputSize(int inputSize, int outputSize) {
		return new NeuronLayer(inputSize, outputSize);
	}

	public static NeuronLayer defaultWithWeights(float[][] layerWeights) {
		return new NeuronLayer(layerWeights);
	}

	private final int inputSize;

	@Getter
	private Neuron[] neurons;

	private final int outputSize;

	@Getter
	private float[] lastOutput;
	
	@Getter
	private float[] lastInput;

	public NeuronLayer(float[][] layerWeights) {
		this(layerWeights, Functions.linearFunction());
	}

	public NeuronLayer(int inputSize, int outputSize) {
		this(inputSize, outputSize, Functions.linearFunction());
	}

	public NeuronLayer(int inputSize, int outputSize, ActivationFunction function) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.function = function;
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = new Neuron(inputSize, function);
		}
	}

	public NeuronLayer(float[][] layerWeights, ActivationFunction function) {
		outputSize = layerWeights.length;
		inputSize = layerWeights[0].length;
		this.function = function;
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = new Neuron(layerWeights[i], function);
		}
	}

	@Override
	public float[] apply(float[] input) {
		lastInput = input;
		lastOutput = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			lastOutput[i] = neurons[i].apply(input);
		}
		return lastOutput;
	}

	public int inputSize() {
		return inputSize;
	}

	public int outputSize() {
		return outputSize;
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

	public void switchFunction(ActivationFunction newFunction) {
		final float[][] weights = weights();
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = new Neuron(weights[i], newFunction);
		}
	}

}
