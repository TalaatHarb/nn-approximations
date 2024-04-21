package net.talaatharb.nn.functions;

public class Tanh implements ActivationFunction {

	@Override
	public Float apply(Float input) {
		return (float) Math.tanh(input);
	}

	@Override
	public float numericalDervative(float input, float output) {
		return 1 - output * output;
	}

}
