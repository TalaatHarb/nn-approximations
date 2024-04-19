package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class ExponentialLinearUnit implements UnaryOperator<Float> {

	private static final float ALPHA = 1.0f;

	@Override
	public Float apply(Float t) {
		return t > 0 ? t : (float) (ALPHA * (Math.exp(t) - 1));
	}

}
