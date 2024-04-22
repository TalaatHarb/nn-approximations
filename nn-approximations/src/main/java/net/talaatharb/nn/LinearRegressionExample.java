package net.talaatharb.nn;

import java.util.List;
import java.util.Random;

import lombok.extern.slf4j.Slf4j;
import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.functions.Functions;
import net.talaatharb.nn.metrics.MeanSquareError;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.TrainingData;

@Slf4j
public class LinearRegressionExample {
	
	private static final Random RANDOM = new Random();
	public static void main(String[] args) {
		final int n = 100;
		final var inputs = new float[n][];
		final var outputs = new float[n][];
		for (int i = 0; i < n ; i++) {
			final float value = i * RANDOM.nextFloat() - i / 2.0f + 0.5f;
			inputs[i] = new float[] {value};
			outputs[i] = new float[] {secretLinearFunction(value)};
		}
		
		trainNetworkOnData(new TrainingData(inputs, outputs));
	}
	
	private static float secretLinearFunction(float x) {
		return 4.267f + 1.373f * x;
	}

	private static void trainNetworkOnData(final TrainingData data) {
		final int[] shape = new int[] { 1, 1 };
		final var functions = List.of(Functions.linearFunction());
		final var network = new FeedForwardNN(shape, functions);
		network.randomize();

		final var algorithm = new StochasticGradientDescent(0.001f);
		final var lossFunction = new MeanSquareError();
		algorithm.train(network, data, 0.000001f, 500, lossFunction);

		log.info(network.getLayers()[0].getNeurons()[0].toString());
	}
}
