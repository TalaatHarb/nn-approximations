package net.talaatharb.nn.structures;

import java.util.function.UnaryOperator;

public class FeedForwardNN implements UnaryOperator<float[]> {

	private final int[] shape;

	public FeedForwardNN(int[] shape) {
		this.shape = shape;
	}

	@Override
	public float[] apply(float[] t) {
		return null;
	}

	public static final FeedForwardNN defaultWithShape(int[] shape) {
		return new FeedForwardNN(shape);
	}

	public int[] shape() {
		return shape;
	}

}
