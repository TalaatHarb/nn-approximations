package net.talaatharb.nn.metrics;

public class MeanSquareError implements LossFunction {

	@Override
	public float errorMetric(float[] expectedOutput, float[] output) {
		final int n = Math.min(expectedOutput.length, output.length);
		float sum = 0.0f;
		for (int i = 0; i < n; i++) {
			sum += Math.pow(output[i] - expectedOutput[i]  , 2);
		}

		return sum / n;
	}

	@Override
	public float[] numericalDervative(float[] expectedOutput, float[] output) {
		int n = Math.min(output.length, expectedOutput.length);
		float[] result = new float[n];
		
		for(int i = 0; i < n; i++) {
			result[i] = 2.0f * (output[i] - expectedOutput[i]) / n;
		}
		
		return result;
	}

}
