package net.talaatharb.nn.metrics;

public interface LossFunction {

	// https://www.analyticsvidhya.com/blog/2022/06/understanding-loss-function-in-deep-learning/#:~:text=Mean%20Squared%20Error%2FSquared%20loss,it%20across%20the%20entire%20dataset.
	float errorMetric(float[] expectedOutput, float[] output);
	
	float[] numericalDervative(float[] expectedOutput, float[] output);
}
