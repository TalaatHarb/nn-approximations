package net.talaatharb.nn.algorithms;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.jupiter.api.Test;

import net.talaatharb.nn.functions.Functions;
import net.talaatharb.nn.metrics.MeanSquareError;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.TrainingData;

class StochasticGradientDescentTest {

	private final MeanSquareError loss = new MeanSquareError();

	@Test
	void updateMatchesFiniteDifferenceGradientAcrossMultipleHiddenLayersAndOutputs() {
		final var network = new FeedForwardNN(new int[] { 2, 3, 2, 2 },
				List.of(Functions.tanhFunction(), Functions.sigmoidFunction(), Functions.linearFunction()));
		network.randomize(new Random(7));
		float[] input = { 0.3f, -0.7f };
		float[] expected = { 0.2f, -0.4f };
		final var before = copy(network);
		final var gradients = copy(network);
		float epsilon = 0.001f;
		for (int l = 0; l < before.length; l++) {
			for (int n = 0; n < before[l].length; n++) {
				for (int w = 0; w < before[l][n].length; w++) {
					network.getLayers()[l].weights()[n][w] = before[l][n][w] + epsilon;
					float plus = loss.errorMetric(expected, network.apply(input));
					network.getLayers()[l].weights()[n][w] = before[l][n][w] - epsilon;
					float minus = loss.errorMetric(expected, network.apply(input));
					gradients[l][n][w] = (plus - minus) / (2 * epsilon);
					network.getLayers()[l].weights()[n][w] = before[l][n][w];
				}
			}
		}
		final var updates = new ArrayList<TrainingUpdate>();
		float rate = 0.05f;
		final var data = new TrainingData(new float[][] { input }, new float[][] { expected });
		float result = new StochasticGradientDescent(rate).train(network, data, 0, 1, loss,
				new TrainingListener() {
					@Override
					public void onUpdate(TrainingUpdate update) {
						updates.add(update);
					}
				}, () -> false);
		for (int l = 0; l < before.length; l++) {
			for (int n = 0; n < before[l].length; n++) {
				for (int w = 0; w < before[l][n].length; w++) {
					assertEquals(before[l][n][w] - rate * gradients[l][n][w],
							network.getLayers()[l].weights()[n][w], 0.00003,
							"Incorrect gradient at layer " + l + ", unit " + n + ", weight " + w);
				}
			}
		}
		assertEquals(1, updates.size());
		final var update = updates.getFirst();
		assertEquals(1, update.epoch());
		assertEquals(1, update.step());
		assertEquals(0, update.sampleIndex());
		assertEquals(List.of(0.2f, -0.4f), update.expected());
		assertTrue(update.lossAfter() < update.lossBefore());
		assertEquals(loss.errorMetric(expected, network.apply(input)), result, 1e-7);
		assertEquals(update.lossAfter(), result, 1e-7);
		assertEquals(network.inspect(input), update.after());
	}

	@Test
	void cancellationRetainsExactlyTheCompletedUpdates() {
		final var network = new FeedForwardNN(new int[] { 1, 1 });
		final var data = new TrainingData(new float[][] { { 1 }, { 2 } }, new float[][] { { 1 }, { 2 } });
		final var stopped = new AtomicBoolean();
		final var updates = new ArrayList<TrainingUpdate>();
		assertThrows(CancellationException.class, () -> new StochasticGradientDescent(0.1f).train(
				network, data, 0, 5, loss, new TrainingListener() {
					@Override
					public void onUpdate(TrainingUpdate update) {
						updates.add(update);
						stopped.set(true);
					}
				}, stopped::get));
		assertEquals(1, updates.size());
		float[] input = { updates.getFirst().after().layers().getFirst().units().getFirst().activation() };
		assertEquals(updates.getFirst().after(), network.inspect(input));
	}

	@Test
	void observerExceptionsAreNotSwallowed() {
		final var network = new FeedForwardNN(new int[] { 1, 1 });
		final var data = new TrainingData(new float[][] { { 1 } }, new float[][] { { 1 } });
		final var failure = new IllegalStateException("observer failed");
		assertSame(failure, assertThrows(IllegalStateException.class,
				() -> new StochasticGradientDescent(0.1f).train(network, data, 0, 1, loss,
						new TrainingListener() {
							@Override
							public void onUpdate(TrainingUpdate update) {
								throw failure;
							}
						}, () -> false)));
	}

	@Test
	void rejectsBadTrainingParametersAndDimensionsBeforeMutating() {
		final var network = new FeedForwardNN(new int[] { 1, 1 });
		final var algorithm = new StochasticGradientDescent(0.1f);
		final var data = new TrainingData(new float[][] { { 1 } }, new float[][] { { 1 } });
		assertThrows(IllegalArgumentException.class, () -> new StochasticGradientDescent(Float.NaN));
		assertThrows(IllegalArgumentException.class, () -> new StochasticGradientDescent(0));
		assertThrows(IllegalArgumentException.class, () -> algorithm.train(network, data, -1, 1, loss));
		assertThrows(IllegalArgumentException.class, () -> algorithm.train(network, data, 0, 0, loss));
		assertThrows(IllegalArgumentException.class, () -> algorithm.train(network,
				new TrainingData(new float[0][], new float[0][]), 0, 1, loss));
		assertThrows(IllegalArgumentException.class, () -> algorithm.train(network,
				new TrainingData(new float[][] { { 1, 2 } }, new float[][] { { 1 } }), 0, 1, loss));
		assertThrows(IllegalArgumentException.class, () -> algorithm.train(network,
				new TrainingData(new float[][] { { Float.NaN } }, new float[][] { { 1 } }), 0, 1, loss));
		assertEquals(0, network.apply(new float[] { 1 })[0]);
	}

	private static float[][][] copy(FeedForwardNN network) {
		final var result = new float[network.getDepth()][][];
		for (int l = 0; l < result.length; l++) {
			final var weights = network.getLayers()[l].weights();
			result[l] = new float[weights.length][];
			for (int n = 0; n < weights.length; n++) {
				result[l][n] = weights[n].clone();
			}
		}
		return result;
	}
}
