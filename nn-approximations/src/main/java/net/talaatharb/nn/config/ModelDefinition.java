package net.talaatharb.nn.config;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import net.talaatharb.nn.structures.FeedForwardNN;

/**
 * Serializable description of an example: network topology, initialization, training settings,
 * dataset and (optionally) trained weights. {@code modelType} leaves room for other model kinds
 * such as RBMs; only {@value #FEED_FORWARD} is supported today.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({ "modelType", "formatVersion", "name", "description", "task", "network", "initialization",
		"training", "dataset", "weights" })
public record ModelDefinition(String modelType, int formatVersion, String name, String description, Task task,
		NetworkSpec network, InitializationSpec initialization, TrainingSpec training, DatasetSpec dataset,
		float[][][] weights) {

	public static final String FEED_FORWARD = "feedForward";
	public static final int FORMAT_VERSION = 1;

	public enum Task {
		/** Continuous targets. */
		@JsonProperty("regression")
		REGRESSION,
		/** Targets in {0, 1}; predictions are thresholded at 0.5 for display. */
		@JsonProperty("classification")
		CLASSIFICATION,
		/** Images in, one-hot class out; the predicted class is the strongest output. */
		@JsonProperty("imageClassification")
		IMAGE_CLASSIFICATION;

		@Override
		public String toString() {
			return switch (this) {
				case REGRESSION -> "Regression";
				case CLASSIFICATION -> "Classification";
				case IMAGE_CLASSIFICATION -> "Image classification";
			};
		}
	}

	public ModelDefinition {
		modelType = modelType == null ? FEED_FORWARD : modelType;
		formatVersion = formatVersion == 0 ? FORMAT_VERSION : formatVersion;
		description = description == null ? "" : description;
		task = task == null ? Task.REGRESSION : task;
		initialization = initialization == null ? InitializationSpec.defaults() : initialization;
	}

	public List<String> validate() {
		final var errors = new ArrayList<String>();
		if (!FEED_FORWARD.equals(modelType)) {
			errors.add("modelType '" + modelType + "' is not supported yet (expected '" + FEED_FORWARD + "')");
		}
		if (formatVersion != FORMAT_VERSION) {
			errors.add("formatVersion " + formatVersion + " is not supported (expected " + FORMAT_VERSION + ")");
		}
		if (name == null || name.isBlank()) {
			errors.add("name is required");
		}
		if (network == null) {
			errors.add("network is required");
		} else {
			network.validate(errors);
		}
		initialization.validate(errors);
		if (training == null) {
			errors.add("training is required");
		} else {
			training.validate(errors);
		}
		if (dataset == null) {
			errors.add("dataset is required");
		} else {
			dataset.validate(errors);
			if (task == Task.IMAGE_CLASSIFICATION != dataset instanceof HuggingFaceDataset) {
				errors.add(task == Task.IMAGE_CLASSIFICATION
						? "task imageClassification needs an image dataset (type huggingFace)"
						: "image datasets (type huggingFace) need task imageClassification");
			}
			if (dataset instanceof HuggingFaceTableDataset table) {
				if (table.isCategorical() && task != Task.CLASSIFICATION) {
					errors.add("a categorical target (dataset.classes) needs task classification");
				}
				if (task == Task.CLASSIFICATION && !table.isCategorical()
						&& table.targetScaling() != DataScaling.Mode.NONE) {
					errors.add("classification with a numeric 0/1 target needs dataset.targetScaling none"
							+ " (or list dataset.classes)");
				}
			}
			if (network != null && !network.layers().isEmpty()) {
				if (dataset.inputSize() != network.inputs()) {
					errors.add("dataset has " + dataset.inputSize() + " input(s) but network.inputs is "
							+ network.inputs());
				}
				final var output = network.layers().get(network.layers().size() - 1);
				if (output != null && dataset.outputSize() != output.neurons()) {
					errors.add("dataset has " + dataset.outputSize() + " output(s) but the output layer has "
							+ output.neurons() + " neuron(s)");
				}
			}
		}
		if (weights != null && network != null && errors.isEmpty()) {
			validateWeights(errors);
		}
		return errors;
	}

	private void validateWeights(List<String> errors) {
		final int[] shape = network.shape();
		if (weights.length != shape.length - 1) {
			errors.add("weights must have one entry per layer (" + (shape.length - 1) + ")");
			return;
		}
		for (int l = 0; l < weights.length; l++) {
			if (weights[l] == null || weights[l].length != shape[l + 1]) {
				errors.add("weights[" + l + "] must have " + shape[l + 1] + " neuron rows");
				return;
			}
			for (float[] row : weights[l]) {
				if (row == null || row.length != shape[l] + 1) {
					errors.add("weights[" + l + "] rows must be [bias, " + shape[l] + " weight(s)]");
					return;
				}
				for (float value : row) {
					if (!Float.isFinite(value)) {
						errors.add("weights[" + l + "] contains a non-finite value");
						return;
					}
				}
			}
		}
	}

	public ModelDefinition requireValid() {
		final var errors = validate();
		if (!errors.isEmpty()) {
			throw new InvalidModelException(errors);
		}
		return this;
	}

	/** Builds the network with stored weights if present, otherwise initialized weights. */
	public FeedForwardNN createNetwork(long seedIfUnspecified) {
		final var net = new FeedForwardNN(network.shape(), network.functions());
		if (weights != null) {
			net.setWeights(weights);
		} else {
			initialization.initialize(net, effectiveSeed(seedIfUnspecified));
		}
		return net;
	}

	public long effectiveSeed(long seedIfUnspecified) {
		return initialization.seed() == null ? seedIfUnspecified : initialization.seed();
	}

	@JsonIgnore
	public boolean isClassification() {
		return task == Task.CLASSIFICATION;
	}

	@JsonIgnore
	public boolean isImageClassification() {
		return task == Task.IMAGE_CLASSIFICATION;
	}

	public ModelDefinition withName(String newName) {
		return new ModelDefinition(modelType, formatVersion, newName, description, task, network, initialization,
				training, dataset, weights);
	}

	public ModelDefinition withWeights(float[][][] newWeights) {
		return new ModelDefinition(modelType, formatVersion, name, description, task, network, initialization,
				training, dataset, newWeights);
	}

	public ModelDefinition withInitialization(InitializationSpec newInitialization) {
		return new ModelDefinition(modelType, formatVersion, name, description, task, network,
				Objects.requireNonNull(newInitialization), training, dataset, weights);
	}

	public static final class InvalidModelException extends IllegalArgumentException {
		private static final long serialVersionUID = 1L;
		private final transient List<String> errors;

		public InvalidModelException(List<String> errors) {
			super("Invalid model:\n - " + String.join("\n - ", errors));
			this.errors = List.copyOf(errors);
		}

		public List<String> errors() {
			return errors;
		}
	}
}
