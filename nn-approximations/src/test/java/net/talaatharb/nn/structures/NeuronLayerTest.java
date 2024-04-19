package net.talaatharb.nn.structures;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.Random;

import org.junit.jupiter.api.Test;

import net.talaatharb.nn.functions.Functions;

class NeuronLayerTest {

	private static final Random RANDOM = new Random();

	@Test
	void testDefaultLayerCanBeCreatedInAnySize() {
		final int inputSize = RANDOM.nextInt(0, 100);
		final int outputSize = RANDOM.nextInt(0, 100);

		final NeuronLayer layer = NeuronLayer.defaultWithInputOutputSize(inputSize, outputSize);

		assertNotNull(layer);
		assertEquals(inputSize, layer.inputSize());
		assertEquals(outputSize, layer.outputSize());
	}

	@Test
	void testRandomizingWeightsOfALayerWorks() {
		final int inputSize = 2;
		final int outputSize = 1;

		final NeuronLayer layer = NeuronLayer.defaultWithInputOutputSize(inputSize, outputSize);
		layer.randomize();

		final float[] result = layer.apply(new float[] { 0.0f, 1.0f });

		assertNotEquals(0.0f, result[0]);

	}

	@Test
	void testSetLayerWeights() {
		final int inputSize = 2;
		final int outputSize = 1;
		final NeuronLayer layer = NeuronLayer.defaultWithInputOutputSize(inputSize, outputSize);
		final float[][] layerWeights = new float[][] { new float[] { 0.5f, 0.5f, 0.5f } };
		layer.setWeights(layerWeights);

		assertArrayEquals(layerWeights, layer.weights());
	}

	@Test
	void testDefaultLayerCanBeCreatedWithInitialWeights() {
		final float[][] layerWeights = new float[][] { new float[] { 0.5f, 0.5f, 0.5f } };

		final NeuronLayer layer = NeuronLayer.defaultWithWeights(layerWeights);

		assertNotNull(layer);
		assertArrayEquals(layerWeights, layer.weights());
	}

	@Test
	void testNeuronLayerCanBeCreatedWithCustomFunction() throws Exception {
		final float[][] layerWeights = new float[][] { new float[] { 0.25f, 0.5f, 0.5f } };
		final NeuronLayer layer = new NeuronLayer(layerWeights, Functions.stepWithThreshold(0.0f));

		final float[] input = new float[] { -0.5f, -0.5f };
		final float[] result = layer.apply(input);

		assertEquals(0.0f, result[0]);
	}

}
