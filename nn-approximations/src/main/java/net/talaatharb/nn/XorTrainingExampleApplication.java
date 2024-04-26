package net.talaatharb.nn;

import java.util.List;

import lombok.extern.slf4j.Slf4j;
import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.functions.Functions;
import net.talaatharb.nn.metrics.MeanSquareError;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.TrainingData;

@Slf4j
public class XorTrainingExampleApplication {

	private static final String LINE = "------------";

	public static void main(String[] args) {
		final var inputs = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 },
				new float[] { 1, 1 } };

		final var xorOutputs = new float[][] { new float[] { 0 }, new float[] { 1 }, new float[] { 1 },
				new float[] { 0 } };

		trainNetworkOnData(new TrainingData(inputs, xorOutputs), "XOR");

	}

	private static void trainNetworkOnData(final TrainingData data, String symbol) {
		final int[] shape = new int[] { 2, 2, 1 };
		final var functions = List.of(Functions.sigmoidFunction(), Functions.sigmoidFunction());
		final var network = new FeedForwardNN(shape, functions);

		final var algorithm = new StochasticGradientDescent(5.0f);
		final var lossFunction = new MeanSquareError();
		final float errorThreshold = 0.01f;
		float error = Float.MAX_VALUE;
		
		// iterating to make a stable network even with gradient vanishing functions like sigmoid
		while (error > errorThreshold) {
			network.randomize();
			error = algorithm.train(network, data, errorThreshold, 30000, lossFunction);
		}
		network.switchLastLayerFunction(Functions.stepWithThreshold(0.0f));
		printTruthTableBinary(network, symbol);
	}

	static final void printTruthTableBinary(FeedForwardNN network, String symbol) {
		final var inputs = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 },
				new float[] { 1, 1 } };
		log.info(LINE);
		log.info(" a | b |" + symbol);
		for (final float[] input : inputs) {
			log.info(input[0] + "|" + input[1] + "|" + network.apply(input)[0]);
		}
		log.info(LINE);
		log.info(network.toString());
	}

}
