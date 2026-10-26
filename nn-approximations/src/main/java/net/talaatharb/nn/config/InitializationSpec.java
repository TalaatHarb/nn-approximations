package net.talaatharb.nn.config;

import java.util.List;
import java.util.Random;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import net.talaatharb.nn.structures.FeedForwardNN;

/**
 * Weight initialization. {@code min}/{@code max} apply to the uniform scheme only;
 * a missing seed means a fresh random seed every time weights are generated.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record InitializationSpec(Scheme scheme, Float min, Float max, Long seed) {

	public enum Scheme {
		/** Biases and weights drawn from [min, max). */
		@JsonProperty("uniform")
		UNIFORM("Uniform [min, max)"),
		/** Glorot/Xavier uniform, zero biases; suits tanh/sigmoid. */
		@JsonProperty("xavierUniform")
		XAVIER_UNIFORM("Xavier/Glorot uniform"),
		/** He/Kaiming uniform, zero biases; suits ReLU-like activations. */
		@JsonProperty("heUniform")
		HE_UNIFORM("He/Kaiming uniform");

		private final String label;

		Scheme(String label) {
			this.label = label;
		}

		@Override
		public String toString() {
			return label;
		}
	}

	public InitializationSpec {
		scheme = scheme == null ? Scheme.UNIFORM : scheme;
		min = min == null ? -1.0f : min;
		max = max == null ? 1.0f : max;
	}

	public static InitializationSpec defaults() {
		return new InitializationSpec(Scheme.UNIFORM, -1.0f, 1.0f, null);
	}

	public InitializationSpec withSeed(Long newSeed) {
		return new InitializationSpec(scheme, min, max, newSeed);
	}

	void validate(List<String> errors) {
		if (scheme == Scheme.UNIFORM && !(Float.isFinite(min) && Float.isFinite(max) && min < max)) {
			errors.add("initialization: uniform requires finite min < max");
		}
	}

	/** Overwrites all weights. Deterministic for a given seed and topology. */
	public void initialize(FeedForwardNN network, long effectiveSeed) {
		final var random = new Random(effectiveSeed);
		for (var layer : network.getLayers()) {
			final int fanIn = layer.inputSize();
			final int fanOut = layer.outputSize();
			final float limit = switch (scheme) {
				case UNIFORM -> 0;
				case XAVIER_UNIFORM -> (float) Math.sqrt(6.0 / (fanIn + fanOut));
				case HE_UNIFORM -> (float) Math.sqrt(6.0 / fanIn);
			};
			final var weights = new float[fanOut][fanIn + 1];
			for (int n = 0; n < fanOut; n++) {
				// Order (bias, then inputs) matches NeuronLayer.randomize for reproducible seeds.
				weights[n][0] = scheme == Scheme.UNIFORM ? random.nextFloat(min, max) : 0;
				for (int i = 1; i <= fanIn; i++) {
					weights[n][i] = scheme == Scheme.UNIFORM ? random.nextFloat(min, max)
							: random.nextFloat(-limit, limit);
				}
			}
			layer.setWeights(weights);
		}
	}
}
