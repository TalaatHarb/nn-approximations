package net.talaatharb.nn.structures;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import net.talaatharb.nn.functions.Functions;

class NeuronTest {

	private static final Random RANDOM = new Random();

	@Test
	void testCreateDefaultNeuronWithAnySize() {
		final int n = RANDOM.nextInt(0, 100);
		final Neuron neuron = Neuron.defaultWithSize(n);

		assertNotNull(neuron);
		assertEquals(n, neuron.size());
	}

	@Test
	void testThroughIllegalArgumentWhenSizeIsNotPositive() {
		final int n = -5;
		final Executable executable = () -> Neuron.defaultWithSize(n);

		assertThrows(IllegalArgumentException.class, executable);
	}

	@Test
	void testCreateDefaultWithWeightsWorks() {
		final float[] weights = new float[] { 0.0f, 0.5f, 0.5f };
		final Neuron neuron = Neuron.defaultWithWeights(weights);

		assertNotNull(neuron);
		assertEquals(weights.length - 1, neuron.size());
	}

	@Test
	void testDefaultNeuronsEvaluateAsLinearDotProductByDefault() {
		final float[] weights = new float[] { 0.5f, 0.5f, 0.5f }; // bias and two weights for two inputs
		final Neuron neuron = Neuron.defaultWithWeights(weights);
		final float[] input = new float[] { 0.5f, 0.5f };
		final float result = neuron.apply(input);

		assertEquals(1.0f, result);
	}

	@Test
	void testNeuronWeightsCanBeUpdated() {
		final float[] originalWeights = new float[] { 0.5f, 0.5f, 0.5f }; // bias and two weights for two inputs
		final Neuron neuron = Neuron.defaultWithWeights(originalWeights);

		final float[] newWeights = new float[] {0.25f, 0.25f, 0.25f};
		neuron.updateWeights(newWeights);

		assertArrayEquals(newWeights, neuron.weights());

		final float newWeightValue = 0.35f;
		neuron.updateWeight(1, newWeightValue); // modifying weight for input 1
		assertEquals(newWeightValue, neuron.weight(1));

		final float newBiasValue = 0.12f;
		neuron.updateBias(newBiasValue);
		assertEquals(newBiasValue, neuron.bias());

	}

	@Test
	void testNeuronsCanBeCreatedWithCustomFunction() throws Exception {
		final float[] weights = new float[] { 0.25f, 0.5f, 0.5f }; // bias and two weights for two inputs
		final Neuron neuron = new Neuron(weights, Functions.stepWithThreshold(0.0f));

		final float[] input = new float[] { -0.5f, -0.5f };
		final float result = neuron.apply(input);

		assertEquals(0.0f, result);
	}

}
