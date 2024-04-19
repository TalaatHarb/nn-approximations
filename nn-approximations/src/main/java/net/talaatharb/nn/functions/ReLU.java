package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class ReLU implements UnaryOperator<Float> {

	@Override
	public Float apply(Float input) {
		return Math.max(0, input);
	}

}
