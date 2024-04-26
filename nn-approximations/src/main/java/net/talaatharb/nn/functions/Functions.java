package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public interface Functions {
	public static final ExponentialLinearUnit ELU = new ExponentialLinearUnit();
	LeakyReLU LEAKY_RELU = new LeakyReLU();
	Linear LINEAR = new Linear();
	Sigmoid SIGMOID = new Sigmoid();

	ReLU RELU = new ReLU();

	Tanh TANH = new Tanh();

	public static ActivationFunction linearFunction() {
		return LINEAR;
	}

	public static ActivationFunction stepWithThreshold(float threshold) {
		return new BinaryStep(threshold);
	}

	public static ActivationFunction piecewiseLinear(float xMin, float xMax) {
		return new PiecewiseLinear(xMin, xMax);
	}

	public static ActivationFunction sigmoidFunction() {
		return SIGMOID;
	}

	public static ActivationFunction tanhFunction() {
		return TANH;
	}

	public static ActivationFunction reluFunction() {
		return RELU;
	}

	public static ActivationFunction leakyReLUFunction() {
		return LEAKY_RELU;
	}

	public static ActivationFunction parametricReLU(float alpha) {
		return new ParametricReLU(alpha);
	}

	public static ActivationFunction exponentialLinearUnitFunction() {
		return ELU;
	}

	public static ActivationFunction scaledExponentialLinearUnit(float alpha) {
		return new ScaledExponentialLinearUnit(alpha);
	}
}
