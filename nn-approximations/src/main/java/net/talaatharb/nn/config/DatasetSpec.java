package net.talaatharb.nn.config;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import net.talaatharb.nn.structures.TrainingData;

/** Training samples, either generated from expressions on a grid or listed explicitly. */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({ @JsonSubTypes.Type(value = FunctionDataset.class, name = "function"),
		@JsonSubTypes.Type(value = TableDataset.class, name = "table") })
public sealed interface DatasetSpec permits FunctionDataset, TableDataset {

	int MAX_SAMPLES = 20_000;

	int inputSize();

	int outputSize();

	List<String> inputNames();

	List<String> outputNames();

	void validate(List<String> errors);

	TrainingData toTrainingData();

	/** [input][0 = min, 1 = max], used to frame visualizations. */
	float[][] inputBounds();
}
