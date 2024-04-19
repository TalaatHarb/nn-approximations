package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public class BinaryStep implements UnaryOperator<Float> {

	private float threshold;

	public BinaryStep(float threshold) {
		this.threshold = threshold;
	}

	@Override
	public Float apply(Float t) {
		return t > threshold ? 1.0f : 0.0f;
	}

}
