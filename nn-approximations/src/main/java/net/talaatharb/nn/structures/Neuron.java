package net.talaatharb.nn.structures;

import java.util.function.Function;

import lombok.ToString;
import net.talaatharb.nn.functions.ActivationFunction;
import net.talaatharb.nn.functions.Functions;

@ToString
public class Neuron implements Function<float[], Float> {

	private final float[] weights;
	private final ActivationFunction function;

	public Neuron(int n) {
		this(n, Functions.linearFunction());
	}

	public Neuron(float[] weights) {
		this(weights, Functions.linearFunction());
	}

	public Neuron(float[] weights, ActivationFunction function) {
		this.weights = weights;
		this.function = function;
	}

	public Neuron(int inputSize, ActivationFunction function) {
		if (inputSize <= 0) {
			throw new IllegalArgumentException("Neurons need positive size");
		}
		weights = new float[inputSize + 1];
		this.function = function;
	}

	public static final Neuron defaultWithSize(int n) {
		return new Neuron(n);
	}

	public int size() {
		return weights.length - 1;
	}

	public static final Neuron defaultWithWeights(float[] weights) {
		return new Neuron(weights);
	}

	@Override
	public Float apply(float[] input) {
		float sum = weights[0];
		final int min = Math.min(input.length, weights.length - 1);
		for (int i = 0; i < min; i++) {
			sum += weights[i + 1] * input[i];
		}
		return function.apply(sum);
	}

	public void updateWeights(float[] newWeights) {
		final int min = Math.min(newWeights.length, weights.length);
		for (int i = 0; i < min; i++) {
			weights[i] = newWeights[i];
		}
	}

	public float[] weights() {
		return weights;
	}

	public void updateWeight(int index, float value) {
		weights[index + 1] = value;
	}

	public void updateBias(float value) {
		weights[0] = value;
	}

	public float weight(int index) {
		return weights[index + 1];
	}

	public float bias() {
		return weights[0];
	}

}
