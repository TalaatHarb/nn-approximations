package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class Tanh implements UnaryOperator<Float> {

	@Override
	public Float apply(Float input) {
		return (float) Math.tanh(input);
	}

}
