package net.talaatharb.nn.config;

import java.util.HashSet;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import net.talaatharb.nn.structures.TrainingData;

/**
 * Samples every combination of evenly spaced input values (a grid) and evaluates one
 * {@link Expression} per output, e.g. {@code "sin(pi*x)"} or {@code "xor(a, b)"}.
 */
public record FunctionDataset(@JsonProperty("inputs") List<Variable> variables, List<String> outputs)
		implements DatasetSpec {

	/** {@code samples} evenly spaced values from min to max inclusive. */
	public record Variable(String name, float min, float max, int samples) {

		@JsonIgnore
		public float value(int index) {
			return samples == 1 ? min : min + (max - min) * index / (samples - 1);
		}
	}

	public FunctionDataset {
		variables = variables == null ? List.of() : List.copyOf(variables);
		outputs = outputs == null ? List.of() : List.copyOf(outputs);
	}

	@Override
	public int inputSize() {
		return variables.size();
	}

	@Override
	public int outputSize() {
		return outputs.size();
	}

	@Override
	public List<String> inputNames() {
		return variables.stream().map(Variable::name).toList();
	}

	@Override
	public List<String> outputNames() {
		return outputs;
	}

	@JsonIgnore
	public long sampleCount() {
		long count = 1;
		for (var variable : variables) {
			count *= Math.max(variable.samples(), 0);
			if (count > Integer.MAX_VALUE) {
				return count;
			}
		}
		return count;
	}

	@Override
	public void validate(List<String> errors) {
		final int before = errors.size();
		if (variables.isEmpty()) {
			errors.add("dataset.inputs needs at least one variable");
		}
		final var names = new HashSet<String>();
		for (var variable : variables) {
			if (variable == null || variable.name() == null || !Expression.IDENTIFIER.matcher(variable.name()).matches()) {
				errors.add("dataset input names must be identifiers such as x or x1");
				continue;
			}
			final String where = "dataset input '" + variable.name() + "'";
			if (Expression.isReserved(variable.name())) {
				errors.add(where + ": name is reserved for a function or constant");
			}
			if (!names.add(variable.name())) {
				errors.add(where + ": duplicate name");
			}
			if (!Float.isFinite(variable.min()) || !Float.isFinite(variable.max())) {
				errors.add(where + ": min and max must be finite");
			} else if (variable.samples() > 1 ? !(variable.min() < variable.max()) : variable.min() > variable.max()) {
				errors.add(where + ": min must be smaller than max");
			}
			if (variable.samples() < 1) {
				errors.add(where + ": samples must be at least 1");
			}
		}
		if (sampleCount() > MAX_SAMPLES) {
			errors.add("dataset: the input grid has " + sampleCount() + " samples; the limit is " + MAX_SAMPLES);
		}
		if (outputs.isEmpty()) {
			errors.add("dataset.outputs needs at least one expression");
		}
		if (errors.size() > before) {
			return;
		}
		try {
			final var data = toTrainingData();
			for (int i = 0; i < data.getOutput().length; i++) {
				for (int o = 0; o < outputs.size(); o++) {
					if (!Float.isFinite(data.getOutput()[i][o])) {
						errors.add("dataset output '" + outputs.get(o) + "' is not finite at "
								+ java.util.Arrays.toString(data.getInput()[i]));
						return;
					}
				}
			}
		} catch (IllegalArgumentException exception) {
			errors.add("dataset: " + exception.getMessage());
		}
	}

	public float[] evaluate(float[] input) {
		final var names = inputNames();
		final double[] values = new double[input.length];
		for (int i = 0; i < input.length; i++) {
			values[i] = input[i];
		}
		final float[] result = new float[outputs.size()];
		for (int o = 0; o < outputs.size(); o++) {
			result[o] = (float) Expression.compile(outputs.get(o), names).evaluate(values);
		}
		return result;
	}

	@Override
	public TrainingData toTrainingData() {
		final var names = inputNames();
		final var expressions = outputs.stream().map(text -> Expression.compile(text, names)).toList();
		final int count = (int) sampleCount();
		final float[][] inputs = new float[count][];
		final float[][] targets = new float[count][];
		final int[] index = new int[variables.size()];
		final double[] values = new double[variables.size()];
		for (int s = 0; s < count; s++) {
			inputs[s] = new float[variables.size()];
			for (int v = 0; v < variables.size(); v++) {
				inputs[s][v] = variables.get(v).value(index[v]);
				values[v] = inputs[s][v];
			}
			targets[s] = new float[expressions.size()];
			for (int o = 0; o < expressions.size(); o++) {
				targets[s][o] = (float) expressions.get(o).evaluate(values);
			}
			// The last variable varies fastest.
			for (int v = variables.size() - 1; v >= 0; v--) {
				if (++index[v] < variables.get(v).samples()) {
					break;
				}
				index[v] = 0;
			}
		}
		return new TrainingData(inputs, targets);
	}

	@Override
	public float[][] inputBounds() {
		return variables.stream().map(v -> new float[] { v.min(), v.max() }).toArray(float[][]::new);
	}
}
