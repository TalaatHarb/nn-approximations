package net.talaatharb.nn.algorithms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.talaatharb.nn.metrics.LossFunction;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.NeuronLayer;
import net.talaatharb.nn.structures.TrainingData;

@Slf4j
@AllArgsConstructor
public class StochasticGradientDescent implements TrainingAlgorithm {

	private float learningRate;

	@Override
	public float train(FeedForwardNN network, TrainingData data, float threshold, int maxEpochs,
			LossFunction lossFunction) {
		log.info("Training statring with error threshold {}, maximum iterations {}, learning rate {}", threshold,
				maxEpochs, learningRate);

		final var inputs = data.getInput();
		final var outputs = data.getOutput();
		final int dataPointsSize = Math.min(inputs.length, outputs.length);
		final List<Integer> pointsList = new ArrayList<>(dataPointsSize);
		for (int i = 0; i < dataPointsSize; i++) {
			pointsList.add(i);
		}

		final int depth = network.getDepth();

		float epochError = 0.0f;
		for (int epoch = 0; epoch < maxEpochs; epoch++) {
			epochError = 0.0f;
			Collections.shuffle(pointsList);
			Integer[] indexes = pointsList.toArray(new Integer[] {});
			for (var dataPointIndex : indexes) {

				// inference
				final var inputVec = inputs[dataPointIndex];
				final var expectedOutput = outputs[dataPointIndex];
				final var outputFromNetwork = network.apply(inputVec);

				// Error
				final float inputError = lossFunction.errorMetric(expectedOutput, outputFromNetwork);
				epochError += inputError;

				// Update step
				float[] layerGradient = lossFunction.numericalDervative(expectedOutput, outputFromNetwork);

				float[] propagatedError = null;
				for (int layerIndex = depth - 1; layerIndex >= 0; layerIndex--) {
					final NeuronLayer currentLayer = network.getLayers()[layerIndex];

					final int numberOfInputsToLayer = currentLayer.inputSize();
					final int numberOfOutputsOfLayer = currentLayer.outputSize();

					final var inputActivation = currentLayer.getLastInput();
					final var outputActivation = currentLayer.getLastOutput();
					final var inputToActivation = currentLayer.getLastOutputBeforeActivation();

					final var layerFunction = currentLayer.getFunction();
					final var layerWeights = currentLayer.weights();

					if (layerIndex != (depth - 1)) {
						layerGradient = propagatedError;
					}
					propagatedError = new float[numberOfInputsToLayer];
					final var updatedWeights = new float[numberOfOutputsOfLayer][];
					for (int k = 0; k < numberOfOutputsOfLayer; k++) {
						layerGradient[k] = layerGradient[k]
								* layerFunction.numericalDervative(inputToActivation[k], outputActivation[k]); // TODO fix
						// input
						updatedWeights[k] = new float[numberOfInputsToLayer + 1];
						for (int m = 0; m < numberOfInputsToLayer + 1; m++) {
							float weightGradient = m == 0 ? 1 : inputActivation[m - 1];
							updatedWeights[k][m] = layerWeights[k][m]
									- learningRate * (layerGradient[k] * weightGradient);
							if (m > 0) {
								propagatedError[m - 1] = layerWeights[k][m - 1] * layerGradient[k];
							}
						}
					}
					currentLayer.setWeights(updatedWeights);
				}
			}
			epochError /= dataPointsSize;
			log.info("EPOCH: {}, error: {}", epoch + 1, epochError);

			if (epochError < threshold) {
				break;
			}

		}
		return epochError;
	}

}
