package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class ParametricReLU implements UnaryOperator<Float> {

	private final float alpha;

	public ParametricReLU(float alpha) {
		this.alpha = alpha;
	}

	@Override
	public Float apply(Float input) {
		return input > 0 ? input : alpha * input;
	}

}
