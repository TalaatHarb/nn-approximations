package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class Linear implements UnaryOperator<Float> {

	@Override
	public Float apply(Float t) {
		return t;
	}

}
