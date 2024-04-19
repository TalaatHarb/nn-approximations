package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public interface Functions {
	public static final ExponentialLinearUnit ELU = new ExponentialLinearUnit();
	LeakyReLU LEAKY_RELU = new LeakyReLU();
	Linear LINEAR = new Linear();
	Sigmoid SIGMOID = new Sigmoid();

	ReLU RELU = new ReLU();

	Tanh TANH = new Tanh();

	public static UnaryOperator<Float> linearFunction() {
		return LINEAR;
	}

	public static UnaryOperator<Float> stepWithThreshold(float threshold) {
		return new BinaryStep(threshold);
	}

	public static UnaryOperator<Float> piecewiseLinear(float xMin, float xMax) {
		return new PiecewiseLinear(xMin, xMax);
	}

	public static UnaryOperator<Float> sigmoidFunction() {
		return SIGMOID;
	}

	public static UnaryOperator<Float> tanhFunction() {
		return TANH;
	}

	public static UnaryOperator<Float> reluFunction() {
		return RELU;
	}

	public static UnaryOperator<Float> leakyReLUFunction() {
		return LEAKY_RELU;
	}

	public static UnaryOperator<Float> parametricReLU(float alpha) {
		return new ParametricReLU(alpha);
	}

	public static UnaryOperator<Float> exponentialLinearUnitFunction() {
		return ELU;
	}

	public static UnaryOperator<Float> scaledExponentialLinearUnit(float alpha) {
		return new ScaledExponentialLinearUnit(alpha);
	}
}
