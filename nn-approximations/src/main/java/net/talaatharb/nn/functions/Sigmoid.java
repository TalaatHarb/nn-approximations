package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class Sigmoid implements UnaryOperator<Float> {

	@Override
	public Float apply(Float input) {
		return 1.0f / (1.0f + (float) Math.exp(-input));
	}

}
