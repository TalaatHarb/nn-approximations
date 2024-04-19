package net.talaatharb.nn.structures;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import net.talaatharb.nn.functions.Functions;

class FeedForwardNNTest {

	private static final Random RANDOM = new Random();

	@Test
	void testDefaultNetworkCanBeCreatedOfAnyStructure() {
		final int numberOfLayersIncludingInput = RANDOM.nextInt(1, 10);
		final int[] shape = new int[numberOfLayersIncludingInput];
		for (int i = 0; i < numberOfLayersIncludingInput; i++) {
			shape[i] = RANDOM.nextInt(1, 10);
		}

		final FeedForwardNN network = FeedForwardNN.defaultWithShape(shape);

		assertNotNull(network);
		assertArrayEquals(shape, network.shape());
	}

	@Test
	void testRandomizingWeightsOfNetwork() {
		final int[] shape = new int[] { 2, 1 };
		final FeedForwardNN network = FeedForwardNN.defaultWithShape(shape);

		network.randomize();

		final float[] result = network.apply(new float[] { 0.0f, 1.0f });

		assertNotEquals(0.0f, result[0]);
	}

	@Test
	void testNetworkCanBeCreatedWithCustomFunctionPerLayer() {
		final int[] shape = new int[] { 2, 1 };
		final var functions = List.of(Functions.stepWithThreshold(0.0f));
		final FeedForwardNN network = new FeedForwardNN(shape, functions);
		final float[][] outputLayerWeights = new float[][] { new float[] { -1.0f, 1.0f, 1.0f } }; // logical 'And'
		final float[][][] networkWeights = new float[][][] { outputLayerWeights };

		network.setWeights(networkWeights);

		final float[] result = network.apply(new float[] { 1.0f, 1.0f });

		assertEquals(1.0f, result[0]);
	}

}
