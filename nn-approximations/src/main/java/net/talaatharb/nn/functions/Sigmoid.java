package net.talaatharb.nn.functions;

public class Sigmoid implements ActivationFunction {

	@Override
	public Float apply(Float input) {
		return 1.0f / (1.0f + (float) Math.exp(-input));
	}

	@Override
	public float numericalDervative(float input, float output) {
		return output * (1 - output);
	}

}
