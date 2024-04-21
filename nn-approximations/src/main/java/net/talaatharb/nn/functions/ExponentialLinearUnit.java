package net.talaatharb.nn.functions;

public class ExponentialLinearUnit implements ActivationFunction {

	private static final float ALPHA = 1.0f;

	@Override
	public Float apply(Float input) {
		return input > 0 ? input : (float) (ALPHA * (Math.exp(input) - 1));
	}

	@Override
	public float numericalDervative(float input, float output) {
		return input > 0 ? 1 : (float) (ALPHA * Math.exp(input));
	}

}
