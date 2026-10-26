package net.talaatharb.nn.config;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import net.talaatharb.nn.structures.TrainingData;

/** Explicit samples, e.g. a truth table. Names are optional and default to x1.. / y1... */
public record TableDataset(List<String> inputNames, List<String> outputNames, float[][] inputs, float[][] outputs)
		implements DatasetSpec {

	public TableDataset {
		inputs = inputs == null ? new float[0][] : inputs;
		outputs = outputs == null ? new float[0][] : outputs;
		inputNames = names(inputNames, inputs, "x");
		outputNames = names(outputNames, outputs, "y");
	}

	private static List<String> names(List<String> names, float[][] rows, String prefix) {
		if (names != null && !names.isEmpty()) {
			return List.copyOf(names);
		}
		final var generated = new ArrayList<String>();
		final int columns = rows.length == 0 || rows[0] == null ? 0 : rows[0].length;
		for (int i = 1; i <= columns; i++) {
			generated.add(prefix + i);
		}
		return List.copyOf(generated);
	}

	@Override
	public int inputSize() {
		return inputNames.size();
	}

	@Override
	public int outputSize() {
		return outputNames.size();
	}

	@Override
	public void validate(List<String> errors) {
		if (inputs.length == 0) {
			errors.add("dataset.inputs needs at least one row");
		}
		if (inputs.length != outputs.length) {
			errors.add("dataset: " + inputs.length + " input rows but " + outputs.length + " output rows");
		}
		if (inputs.length > MAX_SAMPLES) {
			errors.add("dataset: at most " + MAX_SAMPLES + " rows are supported");
		}
		if (new HashSet<>(inputNames).size() != inputNames.size()
				|| new HashSet<>(outputNames).size() != outputNames.size()) {
			errors.add("dataset: input and output names must be unique");
		}
		checkRows("input", inputs, inputNames.size(), errors);
		checkRows("output", outputs, outputNames.size(), errors);
	}

	private static void checkRows(String kind, float[][] rows, int width, List<String> errors) {
		for (int r = 0; r < rows.length; r++) {
			if (rows[r] == null || rows[r].length != width) {
				errors.add("dataset " + kind + " row " + (r + 1) + " must have " + width + " value(s)");
				return;
			}
			for (float value : rows[r]) {
				if (!Float.isFinite(value)) {
					errors.add("dataset " + kind + " row " + (r + 1) + " contains a non-finite value");
					return;
				}
			}
		}
	}

	@Override
	public TrainingData toTrainingData() {
		final float[][] in = new float[inputs.length][];
		final float[][] out = new float[outputs.length][];
		for (int i = 0; i < inputs.length; i++) {
			in[i] = inputs[i].clone();
			out[i] = outputs[i].clone();
		}
		return new TrainingData(in, out);
	}

	@Override
	public float[][] inputBounds() {
		final float[][] bounds = new float[inputSize()][];
		for (int c = 0; c < inputSize(); c++) {
			float min = Float.POSITIVE_INFINITY;
			float max = Float.NEGATIVE_INFINITY;
			for (float[] row : inputs) {
				min = Math.min(min, row[c]);
				max = Math.max(max, row[c]);
			}
			if (!(min < max)) {
				min -= 0.5f;
				max += 0.5f;
			}
			bounds[c] = new float[] { min, max };
		}
		return bounds;
	}
}
