package net.talaatharb.nn.functions;

public class ParametricReLU implements ActivationFunction {

	private final float alpha;

	public ParametricReLU(float alpha) {
		this.alpha = alpha;
	}

	@Override
	public Float apply(Float input) {
		return input > 0 ? input : alpha * input;
	}

	@Override
	public float numericalDervative(float input, float output) {
		if (input == 0) {
			return Float.NaN;
		}

		return input > 0 ? 1 : alpha;
	}

}
