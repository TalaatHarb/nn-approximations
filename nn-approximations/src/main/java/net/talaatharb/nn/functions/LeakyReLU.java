package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class LeakyReLU implements UnaryOperator<Float> {

	private static final float ALPHA = 0.01f;

	@Override
	public Float apply(Float input) {
		return input > 0 ? input : ALPHA * input;
	}

}
