package net.talaatharb.nn.algorithms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;

import lombok.extern.slf4j.Slf4j;
import net.talaatharb.nn.metrics.LossFunction;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.TrainingData;

@Slf4j
public class StochasticGradientDescent implements TrainingAlgorithm {

	private final float learningRate;
	private final Random random;

	public StochasticGradientDescent(float learningRate) {
		this(learningRate, new Random());
	}

	public StochasticGradientDescent(float learningRate, Random random) {
		if (!Float.isFinite(learningRate) || learningRate <= 0) {
			throw new IllegalArgumentException("Learning rate must be finite and positive");
		}
		this.learningRate = learningRate;
		this.random = Objects.requireNonNull(random);
	}

	@Override
	public float train(FeedForwardNN network, TrainingData data, float threshold, int maxEpochs,
			LossFunction lossFunction) {
		return train(network, data, threshold, maxEpochs, lossFunction, TrainingListener.NONE, () -> false);
	}

	/**
	 * Observes every update when a listener is supplied. Cancellation throws rather
	 * than returning a success-shaped result; the last completed update is retained.
	 */
	public float train(FeedForwardNN network, TrainingData data, float threshold, int maxEpochs,
			LossFunction lossFunction, TrainingListener listener, BooleanSupplier cancelled) {
		validate(network, data, threshold, maxEpochs);
		Objects.requireNonNull(lossFunction);
		Objects.requireNonNull(listener);
		Objects.requireNonNull(cancelled);
		final var inputs = data.getInput();
		final var outputs = data.getOutput();
		final List<Integer> indices = new ArrayList<>();
		for (int i = 0; i < inputs.length; i++) {
			indices.add(i);
		}

		long step = 0;
		float epochError = evaluate(network, data, lossFunction);
		for (int epoch = 1; epoch <= maxEpochs && epochError > threshold; epoch++) {
			checkCancellation(cancelled);
			Collections.shuffle(indices, random);
			for (int index : indices) {
				listener.beforeUpdate(epoch, step + 1, index);
				checkCancellation(cancelled);
				final var prediction = network.apply(inputs[index]);
				final float lossBefore = finiteLoss(lossFunction.errorMetric(outputs[index], prediction));
				final boolean observe = listener != TrainingListener.NONE && listener.observeUpdate(step + 1);
				final var before = observe ? network.inspect(inputs[index]) : null;
				float[] gradient = lossFunction.numericalDervative(outputs[index], prediction);
				for (int l = network.getDepth() - 1; l >= 0; l--) {
					final var layer = network.getLayers()[l];
					final var weights = layer.weights();
					final var updated = new float[layer.outputSize()][layer.inputSize() + 1];
					final var propagated = new float[layer.inputSize()];
					for (int n = 0; n < layer.outputSize(); n++) {
						final float delta = gradient[n] * layer.getFunction().numericalDervative(
								layer.getLastOutputBeforeActivation()[n], layer.getLastOutput()[n]);
						updated[n][0] = weights[n][0] - learningRate * delta;
						for (int i = 0; i < layer.inputSize(); i++) {
							// Sum all downstream contributions using the pre-update, non-bias weight.
							propagated[i] += weights[n][i + 1] * delta;
							updated[n][i + 1] = weights[n][i + 1]
									- learningRate * delta * layer.getLastInput()[i];
						}
						for (float weight : updated[n]) {
							if (!Float.isFinite(weight)) {
								throw new ArithmeticException("Training diverged; lower the learning rate");
							}
						}
					}
					layer.setWeights(updated);
					gradient = propagated;
				}
				step++;
				if (observe) {
					final var after = network.inspect(inputs[index]);
					final float lossAfter = finiteLoss(lossFunction.errorMetric(outputs[index],
							network.getLayers()[network.getDepth() - 1].getLastOutput()));
					final var expected = new ArrayList<Float>();
					for (float value : outputs[index]) {
						expected.add(value);
					}
					listener.onUpdate(new TrainingUpdate(epoch, step, index, lossBefore, lossAfter,
							expected, before, after));
				}
			}
			epochError = evaluate(network, data, lossFunction);
			listener.onEpoch(epoch, epochError);
			log.debug("Epoch: {}, loss: {}", epoch, epochError);
		}
		checkCancellation(cancelled);
		return epochError;
	}

	private static float evaluate(FeedForwardNN network, TrainingData data, LossFunction loss) {
		double total = 0;
		for (int i = 0; i < data.getInput().length; i++) {
			total += finiteLoss(loss.errorMetric(data.getOutput()[i], network.apply(data.getInput()[i])));
		}
		return finiteLoss((float) (total / data.getInput().length));
	}

	private static float finiteLoss(float loss) {
		if (!Float.isFinite(loss)) {
			throw new ArithmeticException("Non-finite loss; check data and learning rate");
		}
		return loss;
	}

	private static void checkCancellation(BooleanSupplier cancelled) {
		if (Thread.currentThread().isInterrupted() || cancelled.getAsBoolean()) {
			throw new CancellationException("Training stopped");
		}
	}

	private static void validate(FeedForwardNN network, TrainingData data, float threshold, int maxEpochs) {
		if (!Float.isFinite(threshold) || threshold < 0 || maxEpochs <= 0) {
			throw new IllegalArgumentException("Threshold must be finite and non-negative; epochs must be positive");
		}
		if (data.getInput().length == 0 || data.getInput().length != data.getOutput().length) {
			throw new IllegalArgumentException("Training data must contain matching, non-empty input/output rows");
		}
		final int[] shape = network.shape();
		for (int i = 0; i < data.getInput().length; i++) {
			if (data.getInput()[i].length != shape[0]
					|| data.getOutput()[i].length != shape[shape.length - 1]) {
				throw new IllegalArgumentException("Training row " + i + " does not match network dimensions");
			}
			for (float value : data.getInput()[i]) {
				if (!Float.isFinite(value)) {
					throw new IllegalArgumentException("Training inputs must be finite");
				}
			}
			for (float value : data.getOutput()[i]) {
				if (!Float.isFinite(value)) {
					throw new IllegalArgumentException("Training targets must be finite");
				}
			}
		}
	}
}
