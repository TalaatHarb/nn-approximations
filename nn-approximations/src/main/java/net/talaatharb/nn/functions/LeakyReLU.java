package net.talaatharb.nn.functions;

public class LeakyReLU implements ActivationFunction {

	private static final float ALPHA = 0.01f;

	@Override
	public Float apply(Float input) {
		return input > 0 ? input : ALPHA * input;
	}

	@Override
	public float numericalDervative(float input, float output) {
		if (input == 0) {
			return Float.NaN;
		}

		return input > 0 ? 1 : ALPHA;
	}

}
