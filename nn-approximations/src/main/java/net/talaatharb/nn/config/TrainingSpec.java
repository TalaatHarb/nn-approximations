package net.talaatharb.nn.config;

import java.util.List;
import java.util.Random;

import com.fasterxml.jackson.annotation.JsonInclude;

/** SGD settings. A missing shuffle seed shuffles differently on every run. */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record TrainingSpec(float learningRate, float targetLoss, int maxEpochs, Long shuffleSeed) {

	public static final int MAX_EPOCHS = 1_000_000;

	public Random shuffleRandom() {
		return shuffleSeed == null ? new Random() : new Random(shuffleSeed);
	}

	void validate(List<String> errors) {
		if (!Float.isFinite(learningRate) || learningRate <= 0) {
			errors.add("training.learningRate must be a positive finite number");
		}
		if (!Float.isFinite(targetLoss) || targetLoss < 0) {
			errors.add("training.targetLoss must be a non-negative finite number");
		}
		if (maxEpochs <= 0 || maxEpochs > MAX_EPOCHS) {
			errors.add("training.maxEpochs must be between 1 and " + MAX_EPOCHS);
		}
	}
}
