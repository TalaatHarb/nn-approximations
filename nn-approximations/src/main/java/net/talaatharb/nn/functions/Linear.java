package net.talaatharb.nn.functions;

public class Linear implements ActivationFunction {

	@Override
	public Float apply(Float t) {
		return t;
	}

	@Override
	public float numericalDervative(float input, float output) {
		return 1;
	}

}
