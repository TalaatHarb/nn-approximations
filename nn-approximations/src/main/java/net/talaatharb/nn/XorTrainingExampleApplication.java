package net.talaatharb.nn;

import lombok.extern.slf4j.Slf4j;
import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.config.ExampleCatalog;
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
		final var example = ExampleCatalog.builtIn("xor");
		final var network = example.createNetwork(42);
		final var training = example.training();

		final var algorithm = new StochasticGradientDescent(training.learningRate(), training.shuffleRandom());
		final var lossFunction = new MeanSquareError();
		final float errorThreshold = training.targetLoss();
		final float error = algorithm.train(network, data, errorThreshold, training.maxEpochs(), lossFunction);
		if (error > errorThreshold) {
			throw new IllegalStateException("XOR training did not converge: " + error);
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
