package net.talaatharb.nn;

import java.util.List;

import lombok.extern.slf4j.Slf4j;
import net.talaatharb.nn.functions.Functions;
import net.talaatharb.nn.structures.FeedForwardNN;

@Slf4j
public class NNApproximationApplication {
	public static void main(String[] args) {
		log.info("Application Started");
		final int[] binaryShape = new int[] { 2, 1 };
		final int[] unaryShape = new int[] { 1, 1 };
		final var simpleFunctions = List.of(Functions.stepWithThreshold(0.0f));

		// https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1
		// Basic logical functions
		// logical 'AND'
		final var andWeights = new float[][][] { new float[][] { new float[] { -1.0f, 1.0f, 1.0f } } };
		final var andNetwork = new FeedForwardNN(binaryShape, simpleFunctions);
		andNetwork.setWeights(andWeights);

		// logical 'OR'
		final var orWeights = new float[][][] { new float[][] { new float[] { -1.0f, 2.0f, 2.0f } } };
		final var orNetwork = new FeedForwardNN(binaryShape, simpleFunctions);
		orNetwork.setWeights(orWeights);

		// logical 'NOT'
		final var notWeights = new float[][][] { new float[][] { new float[] { 1.0f, -1.0f } } };
		final var notNetwork = new FeedForwardNN(unaryShape, simpleFunctions);
		notNetwork.setWeights(notWeights);

		// logical 'NOR'
		final var norWeights = new float[][][] { new float[][] { new float[] { 1.0f, -1.0f, -1.0f } } };
		final var norNetwork = new FeedForwardNN(binaryShape, simpleFunctions);
		norNetwork.setWeights(norWeights);

		// logical 'NAND'
		final var nandWeights = new float[][][] { new float[][] { new float[] { 2.0f, -1.0f, -1.0f } } };
		final var nandNetwork = new FeedForwardNN(binaryShape, simpleFunctions);
		nandNetwork.setWeights(nandWeights);

		// A little advanced functions
		final int[] shape = new int[] { 2, 2, 1 };
		final var functions = List.of(Functions.stepWithThreshold(0.0f), Functions.stepWithThreshold(0.0f));

		// logical 'XNOR'
		final var xnorWeights = new float[][][] {
				new float[][] { new float[] { -1.0f, 1.0f, 1.0f }, new float[] { 1.0f, -1.0f, -1.0f } },
				new float[][] { new float[] { -1.0f, 2.0f, 2.0f } } };

		final var xnorNetwork = new FeedForwardNN(shape, functions);
		xnorNetwork.setWeights(xnorWeights);

		// logical 'XOR'
		final var xorWeights = new float[][][] {
				new float[][] { new float[] { -1.0f, 2.0f, 2.0f }, new float[] { 2.0f, -1.0f, -1.0f } },
				new float[][] { new float[] { -1.0f, 1.0f, 1.0f } } };

		final var xorNetwork = new FeedForwardNN(shape, functions);
		xorNetwork.setWeights(xorWeights);

		printTruthTableBinary(andNetwork, "AND");
		printTruthTableBinary(orNetwork, " OR");
		printTruthTableBinary(norNetwork, "NOR");
		printTruthTableBinary(nandNetwork, "NAND");
		printTruthTableBinary(xnorNetwork, "XNOR");
		printTruthTableBinary(xorNetwork, "XOR");
	}

	static final void printTruthTableBinary(FeedForwardNN network, String symbol) {
		final var inputs = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 },
				new float[] { 1, 1 } };
		log.info("------------");
		log.info(" a | b |" + symbol);
		for (final float[] input : inputs) {
			log.info(input[0] + "|" + input[1] + "|" + network.apply(input)[0]);
		}
		log.info("------------");
	}
}