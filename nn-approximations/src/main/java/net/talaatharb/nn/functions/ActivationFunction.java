package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public interface ActivationFunction extends UnaryOperator<Float> {

	// https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions
	float numericalDervative(float input, float output);
	
	default float[] numericalDervative(float[] input, float[] output) {
		int n = Math.min(input.length, output.length);
		float[] result = new float[n];
		for(int i=0; i< n; i++) {
			result[i] = numericalDervative(input[i], output[i]);
		}
		return result;
	}
}
