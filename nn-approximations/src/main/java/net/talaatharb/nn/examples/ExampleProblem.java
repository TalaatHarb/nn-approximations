package net.talaatharb.nn.examples;

import java.util.List;
import java.util.Random;

import net.talaatharb.nn.functions.Functions;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.TrainingData;

/** Reproducible datasets, initial weights and training defaults for the viewer. */
public enum ExampleProblem {
	XOR("XOR", true, 0.3f, 0.005f),
	AND("AND", true, 0.3f, 0.005f),
	OR("OR", true, 0.3f, 0.005f),
	NAND("NAND", true, 0.3f, 0.005f),
	NOR("NOR", true, 0.3f, 0.005f),
	LINEAR("y = 0.3 + 0.7x", false, 0.03f, 0.0001f),
	SQUARE("y = x^2", false, 0.03f, 0.001f),
	SINE("y = sin(pi x)", false, 0.02f, 0.002f);

	private final String label;
	private final boolean logic;
	private final float learningRate;
	private final float threshold;

	ExampleProblem(String label, boolean logic, float learningRate, float threshold) {
		this.label = label;
		this.logic = logic;
		this.learningRate = learningRate;
		this.threshold = threshold;
	}

	public boolean isLogic() {
		return logic;
	}

	public float learningRate() {
		return learningRate;
	}

	public float threshold() {
		return threshold;
	}

	public float target(float... input) {
		float x = input[0];
		return switch (this) {
			case XOR -> x != input[1] ? 1 : 0;
			case AND -> x == 1 && input[1] == 1 ? 1 : 0;
			case OR -> x == 1 || input[1] == 1 ? 1 : 0;
			case NAND -> x == 1 && input[1] == 1 ? 0 : 1;
			case NOR -> x == 1 || input[1] == 1 ? 0 : 1;
			case LINEAR -> 0.3f + 0.7f * x;
			case SQUARE -> x * x;
			case SINE -> (float) Math.sin(Math.PI * x);
		};
	}

	public TrainingData data() {
		int count = logic ? 4 : 41;
		float[][] inputs = new float[count][];
		float[][] outputs = new float[count][];
		for (int i = 0; i < count; i++) {
			inputs[i] = logic ? new float[] { i / 2, i % 2 }
					: new float[] { -1.0f + 2.0f * i / (count - 1) };
			outputs[i] = new float[] { target(inputs[i]) };
		}
		return new TrainingData(inputs, outputs);
	}

	public FeedForwardNN newNetwork() {
		final FeedForwardNN network;
		if (logic) {
			network = new FeedForwardNN(new int[] { 2, 4, 1 },
					List.of(Functions.tanhFunction(), Functions.sigmoidFunction()));
		} else if (this == LINEAR) {
			network = new FeedForwardNN(new int[] { 1, 1 });
		} else {
			network = new FeedForwardNN(new int[] { 1, 8, 1 },
					List.of(Functions.tanhFunction(), Functions.linearFunction()));
		}
		network.randomize(new Random(42));
		return network;
	}

	@Override
	public String toString() {
		return label;
	}
}
