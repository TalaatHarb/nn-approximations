package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class ScaledExponentialLinearUnit implements UnaryOperator<Float> {

	private float alpha;

	public ScaledExponentialLinearUnit(float alpha) {
		this.alpha = alpha;
	}

	@Override
	public Float apply(Float t) {
		return t > 0 ? t : (float) (alpha * (Math.exp(t) - 1));
	}

}
