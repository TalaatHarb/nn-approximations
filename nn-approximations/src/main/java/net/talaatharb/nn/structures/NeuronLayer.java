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
	private ActivationFunction function;

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
	private float[] lastOutputBeforeActivation;
	
	@Getter
	private float[] lastInput;

	public NeuronLayer(float[][] layerWeights) {
		this(layerWeights, Functions.linearFunction());
	}

	public NeuronLayer(int inputSize, int outputSize) {
		this(inputSize, outputSize, Functions.linearFunction());
	}

	public NeuronLayer(int inputSize, int outputSize, ActivationFunction function) {
		if (inputSize <= 0 || outputSize <= 0) {
			throw new IllegalArgumentException("Layer sizes must be positive");
		}
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.function = function;
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = new Neuron(inputSize, function);
		}
	}

	public NeuronLayer(float[][] layerWeights, ActivationFunction function) {
		if (layerWeights.length == 0 || layerWeights[0].length < 2) {
			throw new IllegalArgumentException("Layer weights must include a bias and at least one input");
		}
		outputSize = layerWeights.length;
		inputSize = layerWeights[0].length - 1;
		validateWeights(layerWeights);
		this.function = function;
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = new Neuron(layerWeights[i], function);
		}
	}

	@Override
	public float[] apply(float[] input) {
		if (input.length != inputSize) {
			throw new IllegalArgumentException("Input size does not match layer");
		}
		lastInput = input;
		lastOutput = new float[outputSize];
		lastOutputBeforeActivation = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			Neuron currentNeuron = neurons[i];
			lastOutput[i] = currentNeuron.apply(input);
			lastOutputBeforeActivation[i] = currentNeuron.getLastOutputBeforeActivation();
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
		randomize(RANDOM);
	}

	public void randomize(Random random) {
		for (int i = 0; i < outputSize; i++) {
			final Neuron neuron = neurons[i];
			neuron.updateBias(random.nextFloat(-1.0f, 1.0f));
			for (int j = 0; j < inputSize; j++) {
				neuron.updateWeight(j, random.nextFloat(-1.0f, 1.0f));
			}
		}

	}

	public void setWeights(float[][] layerWeights) {
		validateWeights(layerWeights);
		for (int i = 0; i < outputSize; i++) {
			neurons[i].updateWeights(layerWeights[i]);
		}
	}

	public void validateWeights(float[][] layerWeights) {
		if (layerWeights.length != outputSize) {
			throw new IllegalArgumentException("Weight row count does not match layer");
		}
		for (float[] row : layerWeights) {
			if (row.length != inputSize + 1) {
				throw new IllegalArgumentException("Each weight row must contain bias plus input weights");
			}
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
		this.function = newFunction;
		final float[][] weights = weights();
		neurons = new Neuron[outputSize];
		for (int i = 0; i < outputSize; i++) {
			neurons[i] = new Neuron(weights[i], newFunction);
		}
	}

}
