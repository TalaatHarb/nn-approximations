package net.talaatharb.nn.config;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.json.JsonMapper;

/** Reads and writes {@link ModelDefinition}s as JSON. Arrays stay on one line to keep weights readable. */
public final class ModelJson {

	// Missing numbers become 0 so validate() can report them with a meaningful message.
	private static final ObjectMapper MAPPER = JsonMapper.builder()
			.enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
			.disable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
			.build();

	private static final ObjectWriter WRITER = MAPPER.writer(new DefaultPrettyPrinter()
			.withArrayIndenter(DefaultPrettyPrinter.NopIndenter.instance)
			.withObjectIndenter(new DefaultIndenter("  ", "\n")));

	private ModelJson() {
	}

	/** Parses without semantic validation; call {@link ModelDefinition#validate()} afterwards. */
	public static ModelDefinition read(String json) throws IOException {
		return MAPPER.readValue(json, ModelDefinition.class);
	}

	public static ModelDefinition read(InputStream input) throws IOException {
		return MAPPER.readValue(input, ModelDefinition.class);
	}

	public static ModelDefinition read(Path path) throws IOException {
		return read(Files.readString(path, StandardCharsets.UTF_8));
	}

	public static String write(ModelDefinition model) {
		try {
			return WRITER.writeValueAsString(model) + "\n";
		} catch (JsonProcessingException exception) {
			throw new IllegalStateException("Could not serialize model", exception);
		}
	}

	/** Writes through a temporary file so a failed save never leaves a truncated model. */
	public static void write(ModelDefinition model, Path path) throws IOException {
		final Path parent = path.toAbsolutePath().getParent();
		Files.createDirectories(parent);
		final Path temporary = Files.createTempFile(parent, ".model-", ".tmp");
		try {
			Files.writeString(temporary, write(model), StandardCharsets.UTF_8);
			try {
				Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
			} catch (java.nio.file.AtomicMoveNotSupportedException exception) {
				Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING);
			}
		} finally {
			Files.deleteIfExists(temporary);
		}
	}
}
