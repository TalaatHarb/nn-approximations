package net.talaatharb.nn.export;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import net.talaatharb.nn.config.ModelDefinition;

final class ExportFiles {

	private ExportFiles() {
	}

	static void requireCompatible(ModelDefinition model, float[][][] weights) {
		final int[] shape = model.network().shape();
		if (weights == null || weights.length != shape.length - 1) {
			throw new IllegalArgumentException("Weights do not match the network shape " + model.network().describe());
		}
		for (int l = 0; l < weights.length; l++) {
			if (weights[l].length != shape[l + 1]) {
				throw new IllegalArgumentException("Layer " + (l + 1) + " should have " + shape[l + 1] + " neurons");
			}
			for (float[] neuron : weights[l]) {
				if (neuron.length != shape[l] + 1) {
					throw new IllegalArgumentException(
							"Layer " + (l + 1) + " neurons should have a bias and " + shape[l] + " weights");
				}
				for (float value : neuron) {
					if (!Float.isFinite(value)) {
						throw new IllegalArgumentException("Weights must be finite; reset or retrain the network");
					}
				}
			}
		}
	}

	static void writeAtomically(Path path, byte[] content) throws IOException {
		final Path target = path.toAbsolutePath();
		final Path temp = Files.createTempFile(target.getParent(), target.getFileName().toString(), ".tmp");
		try {
			Files.write(temp, content);
			Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
		} finally {
			Files.deleteIfExists(temp);
		}
	}
}
