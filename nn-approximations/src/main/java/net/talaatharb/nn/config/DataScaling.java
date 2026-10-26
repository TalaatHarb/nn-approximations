package net.talaatharb.nn.config;

import java.util.Arrays;

/**
 * Affine feature/target scaling around the network: {@code network input = (raw - inputOffset) / inputScale}
 * and {@code raw output = network output * outputScale + outputOffset}. Exports embed it so other tools
 * can feed raw column values.
 */
public record DataScaling(float[] inputOffset, float[] inputScale, float[] outputOffset, float[] outputScale) {

	/** How a column is scaled from the training rows. */
	public enum Mode {
		/** (x - mean) / standard deviation. */
		@com.fasterxml.jackson.annotation.JsonProperty("standardize")
		STANDARDIZE,
		/** (x - min) / (max - min), i.e. 0..1 on the training rows. */
		@com.fasterxml.jackson.annotation.JsonProperty("minMax")
		MIN_MAX,
		@com.fasterxml.jackson.annotation.JsonProperty("none")
		NONE;

		@Override
		public String toString() {
			return switch (this) {
				case STANDARDIZE -> "Standardize (z-score)";
				case MIN_MAX -> "Min-max to 0..1";
				case NONE -> "None (raw values)";
			};
		}
	}

	public DataScaling {
		if (inputOffset.length != inputScale.length || outputOffset.length != outputScale.length) {
			throw new IllegalArgumentException("Offsets and scales must have the same length");
		}
		for (float[] values : new float[][] { inputOffset, inputScale, outputOffset, outputScale }) {
			for (float value : values) {
				if (!Float.isFinite(value)) {
					throw new IllegalArgumentException("Scaling values must be finite");
				}
			}
		}
		for (float[] scales : new float[][] { inputScale, outputScale }) {
			for (float scale : scales) {
				if (scale == 0) {
					throw new IllegalArgumentException("Scales must not be zero");
				}
			}
		}
		inputOffset = inputOffset.clone();
		inputScale = inputScale.clone();
		outputOffset = outputOffset.clone();
		outputScale = outputScale.clone();
	}

	public static DataScaling identity(int inputs, int outputs) {
		final float[] ones = new float[inputs];
		final float[] outputOnes = new float[outputs];
		Arrays.fill(ones, 1);
		Arrays.fill(outputOnes, 1);
		return new DataScaling(new float[inputs], ones, new float[outputs], outputOnes);
	}

	/** Offset and scale per column of {@code rows} for the given mode (constant columns get scale 1). */
	public static float[][] fit(float[][] rows, int width, Mode mode) {
		final float[] offset = new float[width];
		final float[] scale = new float[width];
		Arrays.fill(scale, 1);
		if (mode == Mode.NONE || rows.length == 0) {
			return new float[][] { offset, scale };
		}
		for (int c = 0; c < width; c++) {
			double sum = 0;
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			for (float[] row : rows) {
				sum += row[c];
				min = Math.min(min, row[c]);
				max = Math.max(max, row[c]);
			}
			final double mean = sum / rows.length;
			double squares = 0;
			for (float[] row : rows) {
				squares += (row[c] - mean) * (row[c] - mean);
			}
			final double spread = mode == Mode.STANDARDIZE ? Math.sqrt(squares / rows.length) : max - min;
			offset[c] = (float) (mode == Mode.STANDARDIZE ? mean : min);
			scale[c] = spread > 1e-12 && Float.isFinite((float) spread) ? (float) spread : 1;
		}
		return new float[][] { offset, scale };
	}

	public boolean isIdentity() {
		return isIdentity(inputOffset, inputScale) && isIdentity(outputOffset, outputScale);
	}

	public boolean scalesInputs() {
		return !isIdentity(inputOffset, inputScale);
	}

	public boolean scalesOutputs() {
		return !isIdentity(outputOffset, outputScale);
	}

	private static boolean isIdentity(float[] offset, float[] scale) {
		for (int i = 0; i < offset.length; i++) {
			if (offset[i] != 0 || scale[i] != 1) {
				return false;
			}
		}
		return true;
	}

	public float[] scaleInput(float[] raw) {
		return forward(raw, inputOffset, inputScale);
	}

	public float[] unscaleInput(float[] scaled) {
		return backward(scaled, inputOffset, inputScale);
	}

	/** Raw target to the network's output space (for training). */
	public float[] scaleOutput(float[] raw) {
		return forward(raw, outputOffset, outputScale);
	}

	/** Network output back to target units. */
	public float[] unscaleOutput(float[] network) {
		return backward(network, outputOffset, outputScale);
	}

	private static float[] forward(float[] values, float[] offset, float[] scale) {
		final float[] result = new float[values.length];
		for (int i = 0; i < values.length; i++) {
			result[i] = (values[i] - offset[i]) / scale[i];
		}
		return result;
	}

	private static float[] backward(float[] values, float[] offset, float[] scale) {
		final float[] result = new float[values.length];
		for (int i = 0; i < values.length; i++) {
			result[i] = values[i] * scale[i] + offset[i];
		}
		return result;
	}

	@Override
	public boolean equals(Object other) {
		return other instanceof DataScaling that && Arrays.equals(inputOffset, that.inputOffset)
				&& Arrays.equals(inputScale, that.inputScale) && Arrays.equals(outputOffset, that.outputOffset)
				&& Arrays.equals(outputScale, that.outputScale);
	}

	@Override
	public int hashCode() {
		return Arrays.hashCode(new int[] { Arrays.hashCode(inputOffset), Arrays.hashCode(inputScale),
				Arrays.hashCode(outputOffset), Arrays.hashCode(outputScale) });
	}

	@Override
	public String toString() {
		return "DataScaling[inputOffset=" + Arrays.toString(inputOffset) + ", inputScale=" + Arrays.toString(inputScale)
				+ ", outputOffset=" + Arrays.toString(outputOffset) + ", outputScale=" + Arrays.toString(outputScale)
				+ "]";
	}
}
