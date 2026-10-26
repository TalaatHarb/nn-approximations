package net.talaatharb.nn.tracking;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.export.OnnxExporter;
import net.talaatharb.nn.export.PmmlExporter;
import net.talaatharb.nn.tracking.MlflowRunLogger.Artifact;

/**
 * Artifacts logged with a run: the model JSON (with weights), PMML when expressible, and an MLflow model directory
 * {@code model/} with the ONNX flavor so {@code mlflow.onnx.load_model} / {@code mlflow.pyfunc.load_model} work on
 * {@code runs:/<run id>/model}.
 */
public final class RunArtifacts {

	public static final String MODEL_DIRECTORY = "model";

	private RunArtifacts() {
	}

	/** Exports that fail (for example non-finite weights after divergence) are skipped and listed in warnings. */
	public static List<Artifact> build(ModelDefinition model, float[][][] weights, List<String> warnings) {
		final var artifacts = new ArrayList<Artifact>();
		final var trained = model.withWeights(weights);
		artifacts.add(new Artifact("nn-model.json", utf8(ModelJson.write(trained))));
		try {
			final byte[] onnx = OnnxExporter.toBytes(model, weights);
			artifacts.add(new Artifact(MODEL_DIRECTORY + "/model.onnx", onnx));
			artifacts.add(new Artifact(MODEL_DIRECTORY + "/MLmodel", utf8(mlmodel(model))));
			artifacts.add(new Artifact(MODEL_DIRECTORY + "/requirements.txt", utf8("mlflow\nonnx\nonnxruntime\n")));
			artifacts.add(new Artifact(MODEL_DIRECTORY + "/python_env.yaml",
					utf8("python: 3.11\nbuild_dependencies:\n- pip\ndependencies:\n- -r requirements.txt\n")));
			artifacts.add(new Artifact(MODEL_DIRECTORY + "/conda.yaml", utf8("channels:\n- conda-forge\n"
					+ "dependencies:\n- python=3.11\n- pip\n- pip:\n  - mlflow\n  - onnx\n  - onnxruntime\n"
					+ "name: mlflow-env\n")));
		} catch (IllegalArgumentException exception) {
			warnings.add("ONNX model not logged: " + exception.getMessage());
		}
		if (PmmlExporter.unsupportedReason(model).isEmpty()) {
			try {
				artifacts.add(new Artifact("model.pmml", utf8(PmmlExporter.toXml(model, weights))));
			} catch (IllegalArgumentException exception) {
				warnings.add("PMML not logged: " + exception.getMessage());
			}
		}
		return artifacts;
	}

	static String mlmodel(ModelDefinition model) {
		final int inputs = model.network().inputs();
		final int outputs = model.dataset().outputSize();
		return "artifact_path: " + MODEL_DIRECTORY + "\n"
				+ "flavors:\n"
				+ "  onnx:\n"
				+ "    code: null\n"
				+ "    data: model.onnx\n"
				+ "    onnx_version: 1.16.0\n"
				+ "    providers:\n"
				+ "    - CPUExecutionProvider\n"
				+ "  python_function:\n"
				+ "    data: model.onnx\n"
				+ "    env:\n"
				+ "      conda: conda.yaml\n"
				+ "      virtualenv: python_env.yaml\n"
				+ "    loader_module: mlflow.onnx\n"
				+ "    python_version: '3.11'\n"
				+ "model_uuid: " + UUID.randomUUID().toString().replace("-", "") + "\n"
				+ "signature:\n"
				+ "  inputs: '" + tensorSpec(OnnxExporter.INPUT, inputs) + "'\n"
				+ "  outputs: '" + tensorSpec(OnnxExporter.OUTPUT, outputs) + "'\n"
				+ "  params: null\n"
				+ "utc_time_created: '" + Instant.now() + "'\n";
	}

	private static String tensorSpec(String name, int columns) {
		return "[{\"type\": \"tensor\", \"tensor-spec\": {\"dtype\": \"float32\", \"shape\": [-1, " + columns
				+ "]}, \"name\": \"" + name + "\"}]";
	}

	private static byte[] utf8(String text) {
		return text.getBytes(StandardCharsets.UTF_8);
	}
}
