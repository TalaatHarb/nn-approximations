package net.talaatharb.nn.functions;

public class ReLU implements ActivationFunction {

	@Override
	public Float apply(Float input) {
		return Math.max(0, input);
	}

	@Override
	public float numericalDervative(float input, float output) {
		if (input == 0) {
			return Float.NaN;
		}
		return input > 0 ? 1 : 0.0f;
	}

}
