package net.talaatharb.nn.algorithms;

import net.talaatharb.nn.metrics.LossFunction;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.TrainingData;

public interface TrainingAlgorithm {

	float train(FeedForwardNN network, TrainingData data, float threshold, int maxEpochs, LossFunction lossFunction);

}
