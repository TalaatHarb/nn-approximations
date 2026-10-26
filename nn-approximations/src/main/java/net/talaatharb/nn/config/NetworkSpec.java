package net.talaatharb.nn.config;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;

import net.talaatharb.nn.functions.ActivationFunction;

/** Input size plus every non-input layer (the last entry is the output layer). */
public record NetworkSpec(int inputs, List<LayerSpec> layers) {

	public static final int MAX_NEURONS_PER_LAYER = 512;
	public static final int MAX_LAYERS = 32;

	public record LayerSpec(int neurons, ActivationSpec activation) {
	}

	public NetworkSpec {
		layers = layers == null ? List.of() : List.copyOf(layers);
	}

	@JsonIgnore
	public int[] shape() {
		final int[] shape = new int[layers.size() + 1];
		shape[0] = inputs;
		for (int i = 0; i < layers.size(); i++) {
			shape[i + 1] = layers.get(i).neurons();
		}
		return shape;
	}

	@JsonIgnore
	public List<ActivationFunction> functions() {
		return layers.stream().map(layer -> layer.activation().create()).toList();
	}

	@JsonIgnore
	public int parameterCount() {
		int count = 0;
		int previous = inputs;
		for (var layer : layers) {
			count += layer.neurons() * (previous + 1);
			previous = layer.neurons();
		}
		return count;
	}

	/** e.g. "2 → 4 tanh → 1 sigmoid". */
	public String describe() {
		final var parts = new ArrayList<String>();
		parts.add(Integer.toString(inputs));
		for (var layer : layers) {
			parts.add(layer.neurons() + " " + (layer.activation() == null ? "?" : layer.activation()));
		}
		return String.join(" → ", parts);
	}

	void validate(List<String> errors) {
		if (inputs <= 0) {
			errors.add("network.inputs must be positive");
		}
		if (layers.isEmpty()) {
			errors.add("network.layers needs at least the output layer");
		}
		if (layers.size() > MAX_LAYERS) {
			errors.add("network.layers supports at most " + MAX_LAYERS + " layers");
		}
		for (int i = 0; i < layers.size(); i++) {
			final var layer = layers.get(i);
			final String where = i == layers.size() - 1 ? "output layer" : "hidden layer " + (i + 1);
			if (layer == null) {
				errors.add(where + " is missing");
				continue;
			}
			if (layer.neurons() <= 0 || layer.neurons() > MAX_NEURONS_PER_LAYER) {
				errors.add(where + ": neurons must be between 1 and " + MAX_NEURONS_PER_LAYER);
			}
			if (layer.activation() == null) {
				errors.add(where + ": activation is required");
			} else {
				layer.activation().validate(where, errors);
			}
		}
	}
}
