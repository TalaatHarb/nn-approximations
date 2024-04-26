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
public class MultiLinearRegressionExample {

	private static final Random RANDOM = new Random();

	public static void main(String[] args) {
		final int n = 100;
		final var inputs = new float[n][];
		final var outputs = new float[n][];
		for (int i = 0; i < n; i++) {
			final float x1 = i * RANDOM.nextFloat() - i / 2.0f + 0.5f;
			final float x2 = i * RANDOM.nextFloat() - i / 2.0f - 0.5f;
			inputs[i] = new float[] { x1, x2 };
			outputs[i] = new float[] { s1(x1), s2(x2), s3(x1, x2) };
		}

		trainNetworkOnData(new TrainingData(inputs, outputs));
	}

	private static float s1(float x) {
		return 4.267f + 1.373f * x;
	}

	private static float s2(float x) {
		return 5.64f - 3.76f * x;
	}

	private static float s3(float x1, float x2) {
		return 0.23f - 7.54f * x1 + 6.14f * x2;
	}

	private static void trainNetworkOnData(final TrainingData data) {
		final int[] shape = new int[] { 2, 3 };
		final var functions = List.of(Functions.linearFunction());
		final var network = new FeedForwardNN(shape, functions);
		network.randomize();

		final var algorithm = new StochasticGradientDescent(0.001f);
		final var lossFunction = new MeanSquareError();
		algorithm.train(network, data, 0.0000005f, 1000, lossFunction);

		for(int i = 0; i < 3; i++) {			
			log.info(network.getLayers()[0].getNeurons()[i].toString());
		}
	}
}
