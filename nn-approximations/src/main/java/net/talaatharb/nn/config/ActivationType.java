package net.talaatharb.nn.config;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import net.talaatharb.nn.functions.ActivationFunction;
import net.talaatharb.nn.functions.Functions;

/** Activation functions available to JSON models, with their JSON ids and parameters. */
public enum ActivationType {
	LINEAR("linear", "Linear"),
	SIGMOID("sigmoid", "Sigmoid"),
	TANH("tanh", "Tanh"),
	RELU("relu", "ReLU"),
	LEAKY_RELU("leakyRelu", "Leaky ReLU (0.01)"),
	PARAMETRIC_RELU("parametricRelu", "Parametric ReLU", "alpha", 0.1f),
	ELU("elu", "ELU"),
	SCALED_ELU("scaledElu", "ELU with alpha", "alpha", 1.0f),
	PIECEWISE_LINEAR("piecewiseLinear", "Piecewise linear", "xMin", -1.0f, "xMax", 1.0f),
	STEP("step", "Binary step (not trainable)", "threshold", 0.0f);

	private final String id;
	private final String label;
	private final List<String> parameters;
	private final List<Float> defaults;

	ActivationType(String id, String label, Object... parameterDefaults) {
		this.id = id;
		this.label = label;
		final String[] names = new String[parameterDefaults.length / 2];
		final Float[] values = new Float[parameterDefaults.length / 2];
		for (int i = 0; i < names.length; i++) {
			names[i] = (String) parameterDefaults[2 * i];
			values[i] = (Float) parameterDefaults[2 * i + 1];
		}
		this.parameters = List.of(names);
		this.defaults = List.of(values);
	}

	public String id() {
		return id;
	}

	public List<String> parameters() {
		return parameters;
	}

	public float defaultValue(String parameter) {
		return defaults.get(parameters.indexOf(parameter));
	}

	/** False when the derivative is zero almost everywhere, so SGD cannot learn through it. */
	public boolean trainable() {
		return this != STEP;
	}

	public static Optional<ActivationType> fromId(String id) {
		return Arrays.stream(values()).filter(type -> type.id.equals(id)).findFirst();
	}

	ActivationFunction create(Map<String, Float> params) {
		final java.util.function.Function<String, Float> p = name -> params.getOrDefault(name, defaultValue(name));
		return switch (this) {
			case LINEAR -> Functions.linearFunction();
			case SIGMOID -> Functions.sigmoidFunction();
			case TANH -> Functions.tanhFunction();
			case RELU -> Functions.reluFunction();
			case LEAKY_RELU -> Functions.leakyReLUFunction();
			case PARAMETRIC_RELU -> Functions.parametricReLU(p.apply("alpha"));
			case ELU -> Functions.exponentialLinearUnitFunction();
			case SCALED_ELU -> Functions.scaledExponentialLinearUnit(p.apply("alpha"));
			case PIECEWISE_LINEAR -> Functions.piecewiseLinear(p.apply("xMin"), p.apply("xMax"));
			case STEP -> Functions.stepWithThreshold(p.apply("threshold"));
		};
	}

	@Override
	public String toString() {
		return label;
	}
}
