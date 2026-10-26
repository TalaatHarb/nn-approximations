package net.talaatharb.nn.functions;

public class BinaryStep implements ActivationFunction {

	private float threshold;

	public BinaryStep(float threshold) {
		this.threshold = threshold;
	}

	@Override
	public Float apply(Float t) {
		return t > threshold ? 1.0f : 0.0f;
	}

	@Override
	public float numericalDervative(float input, float output) {
		// A step has no useful gradient; gradient-based training cannot adjust weights behind it.
		return 0;
	}

}
