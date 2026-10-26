package net.talaatharb.nn.config;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import net.talaatharb.nn.structures.TrainingData;

/** Training samples: generated from expressions on a grid, listed explicitly, or rows/images from Hugging Face. */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({ @JsonSubTypes.Type(value = FunctionDataset.class, name = "function"),
		@JsonSubTypes.Type(value = TableDataset.class, name = "table"),
		@JsonSubTypes.Type(value = HuggingFaceDataset.class, name = "huggingFace"),
		@JsonSubTypes.Type(value = HuggingFaceTableDataset.class, name = "huggingFaceTable") })
public sealed interface DatasetSpec permits FunctionDataset, TableDataset, HuggingFaceDataset,
		HuggingFaceTableDataset {

	int MAX_SAMPLES = 20_000;

	int inputSize();

	int outputSize();

	List<String> inputNames();

	List<String> outputNames();

	void validate(List<String> errors);

	TrainingData toTrainingData();

	/** [input][0 = min, 1 = max], used to frame visualizations. */
	float[][] inputBounds();

	/** Maps raw values to what the network sees; identity unless the dataset normalizes its columns. */
	default DataScaling scaling() {
		return DataScaling.identity(inputSize(), outputSize());
	}
}
