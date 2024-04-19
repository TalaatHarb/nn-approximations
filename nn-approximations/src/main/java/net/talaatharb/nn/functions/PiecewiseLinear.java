package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class PiecewiseLinear implements UnaryOperator<Float> {

	private final float xMin;
	private final float xMax;
	private final float m;
	private final float b;

	public PiecewiseLinear(float xMin, float xMax) {
		this.xMin = xMin;
		this.xMax = xMax;
		m = 1.0f / (xMax - xMin);
		b = -m * xMin;
	}

	@Override
	public Float apply(Float input) {
		if (input < xMin) {
			return 0.0f;
		}

		if (input <= xMax) {
			return m * input + b;
		}

		return 1.0f;
	}

}
