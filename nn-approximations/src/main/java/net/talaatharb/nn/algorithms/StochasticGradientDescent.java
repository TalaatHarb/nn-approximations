package net.talaatharb.nn.algorithms;

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
	public void train(FeedForwardNN network, TrainingData data, float threshold, int maxEpochs,
			LossFunction lossFunction) {
		log.info("Training statring with error threshold {}, maximum iterations {}, learning rate {}", threshold,
				maxEpochs, learningRate);

		final var inputs = data.getInput();
		final var outputs = data.getOutput();
		final int dataPointsSize = Math.min(inputs.length, outputs.length);
		final int depth = network.getDepth();

		// TODO extend to multi-layer scenario
		for (int epoch = 0; epoch < maxEpochs; epoch++) {

			float epochError = 0.0f;
			// TODO shuffle data points
			for (int dataPointIndex = 0; dataPointIndex < dataPointsSize; dataPointIndex++) {

				// inference
				final var inputVec = inputs[dataPointIndex];
				final var expectedOutput = outputs[dataPointIndex];
				final var outputFromNetwork = network.apply(inputVec);

				// Error
				final float inputError = lossFunction.errorMetric(expectedOutput, outputFromNetwork);
				epochError += inputError;

				// Update step
				final float[] lossFunctionGradient = lossFunction.numericalDervative(expectedOutput, outputFromNetwork);

				for (int layerIndex = depth - 1; layerIndex >= 0; layerIndex--) {
					final NeuronLayer currentLayer = network.getLayers()[layerIndex];

					final int numberOfInputsToLayer = currentLayer.inputSize();
					final int numberOfOutputsOfLayer = currentLayer.outputSize();

					final var inputActivation = currentLayer.getLastInput();
					final var outputActivation = currentLayer.getLastOutput();

					final var layerFunction = currentLayer.getFunction();
					final var layerWeights = currentLayer.weights();

					final var layerGradient = new float[numberOfOutputsOfLayer];
					final var updatedWeights = new float[numberOfOutputsOfLayer][];
					for (int k = 0; k < numberOfOutputsOfLayer; k++) {
						layerGradient[k] = layerFunction.numericalDervative(0.0f, outputActivation[k]); // TODO fix
																										// input
						updatedWeights[k] = new float[numberOfInputsToLayer + 1];
						for (int m = 0; m < numberOfInputsToLayer + 1; m++) {
							float weightGradient = m ==0 ? 1 : inputActivation[m-1];
							updatedWeights[k][m] = layerWeights[k][m]
									- learningRate * lossFunctionGradient[k] * layerGradient[k] * weightGradient;
						}
					}
					currentLayer.setWeights(updatedWeights);
				}

			}
			epochError /= dataPointsSize;
			log.info("EPOCH: {}, error: {}", epoch + 1, epochError);
			
			if(epochError < threshold) break;

		}
	}

}
