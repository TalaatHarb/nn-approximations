package net.talaatharb.nn.functions;

public class ScaledExponentialLinearUnit implements ActivationFunction {

	private float alpha;

	public ScaledExponentialLinearUnit(float alpha) {
		this.alpha = alpha;
	}

	@Override
	public Float apply(Float t) {
		return t > 0 ? t : (float) (alpha * (Math.exp(t) - 1));
	}

	@Override
	public float numericalDervative(float input, float output) {
		return input > 0 ? 1 : (float) (alpha * Math.exp(input));
	}

}
