package net.talaatharb.nn.structures;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.Random;

import org.junit.jupiter.api.Test;

class FeedForwardNNTest {

	private static final Random RANDOM = new Random();

	@Test
	void testDefaultNetworkCanBeCreatedOfAnyStructure() {
		final int depth = RANDOM.nextInt(1, 10);
		final int[] shape = new int[depth];
		for (int i = 0; i < depth; i++) {
			shape[i] = RANDOM.nextInt(1, 10);
		}

		final FeedForwardNN network = FeedForwardNN.defaultWithShape(shape);

		assertNotNull(network);
		assertArrayEquals(shape, network.shape());
	}

}
