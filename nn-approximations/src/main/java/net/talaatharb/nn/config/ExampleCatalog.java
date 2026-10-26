package net.talaatharb.nn.config;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Built-in examples (classpath {@code /examples/<id>.json}) merged with user models stored as
 * {@code <id>.json} in a directory. A user file whose id matches a built-in overrides it;
 * deleting that file restores the built-in.
 */
public final class ExampleCatalog {

	public static final List<String> BUILT_IN_IDS = List.of("xor", "and", "or", "nand", "nor", "linear", "square",
			"sine", "mnist", "iris", "california_housing");
	public static final String MODELS_DIR_PROPERTY = "nn.models.dir";

	public enum Source {
		BUILT_IN("built-in"), MODIFIED_BUILT_IN("modified built-in"), USER("user");

		private final String label;

		Source(String label) {
			this.label = label;
		}

		@Override
		public String toString() {
			return label;
		}
	}

	public record Entry(String id, ModelDefinition model, Source source) {
		@Override
		public String toString() {
			return model.name() + (source == Source.BUILT_IN ? "" : "  (" + source + ")");
		}
	}

	private final Path directory;
	private final Map<String, Entry> entries = new LinkedHashMap<>();
	private final List<String> problems = new ArrayList<>();

	public ExampleCatalog(Path directory) {
		this.directory = directory;
		reload();
	}

	public static Path defaultDirectory() {
		final String configured = System.getProperty(MODELS_DIR_PROPERTY);
		return configured != null && !configured.isBlank() ? Path.of(configured)
				: Path.of(System.getProperty("user.home"), ".nn-approximations", "models");
	}

	public static ExampleCatalog openDefault() {
		return new ExampleCatalog(defaultDirectory());
	}

	public static ModelDefinition builtIn(String id) {
		try (InputStream input = ExampleCatalog.class.getResourceAsStream("/examples/" + id + ".json")) {
			if (input == null) {
				throw new IllegalArgumentException("No built-in example '" + id + "'");
			}
			return ModelJson.read(input).requireValid();
		} catch (IOException exception) {
			throw new UncheckedIOException("Could not read built-in example '" + id + "'", exception);
		}
	}

	public synchronized void reload() {
		entries.clear();
		problems.clear();
		for (String id : BUILT_IN_IDS) {
			entries.put(id, new Entry(id, builtIn(id), Source.BUILT_IN));
		}
		if (!Files.isDirectory(directory)) {
			return;
		}
		final var userEntries = new ArrayList<Entry>();
		try (DirectoryStream<Path> files = Files.newDirectoryStream(directory, "*.json")) {
			for (Path file : files) {
				final String fileName = file.getFileName().toString();
				final String id = fileName.substring(0, fileName.length() - ".json".length());
				if (!isValidId(id)) {
					problems.add(fileName + ": file name must use only a-z, 0-9 and '-'");
					continue;
				}
				try {
					final var model = ModelJson.read(file);
					final var errors = model.validate();
					if (!errors.isEmpty()) {
						problems.add(fileName + ": " + String.join("; ", errors));
						continue;
					}
					userEntries.add(new Entry(id, model,
							BUILT_IN_IDS.contains(id) ? Source.MODIFIED_BUILT_IN : Source.USER));
				} catch (IOException exception) {
					problems.add(fileName + ": " + firstLine(exception.getMessage()));
				}
			}
		} catch (IOException exception) {
			problems.add(directory + ": " + exception.getMessage());
		}
		userEntries.sort(Comparator.comparing(entry -> entry.model().name().toLowerCase(Locale.ROOT)));
		for (var entry : userEntries) {
			entries.put(entry.id(), entry);
		}
	}

	private static String firstLine(String message) {
		if (message == null) {
			return "unreadable";
		}
		final int newline = message.indexOf('\n');
		return newline < 0 ? message : message.substring(0, newline);
	}

	public synchronized List<Entry> entries() {
		return List.copyOf(entries.values());
	}

	public synchronized Optional<Entry> find(String id) {
		return Optional.ofNullable(entries.get(id));
	}

	public synchronized List<String> problems() {
		return List.copyOf(problems);
	}

	public Path directory() {
		return directory;
	}

	public static boolean isValidId(String id) {
		return id != null && id.matches("[a-z0-9]+(-[a-z0-9]+)*");
	}

	/** Turns a display name into a file-safe id that does not collide with any existing entry. */
	public synchronized String newIdFor(String name) {
		String base = name == null ? "" : name.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-")
				.replaceAll("(^-+|-+$)", "");
		if (base.isEmpty()) {
			base = "model";
		}
		if (base.length() > 48) {
			base = base.substring(0, 48).replaceAll("-+$", "");
		}
		String id = base;
		for (int i = 2; entries.containsKey(id) || Files.exists(fileFor(id)); i++) {
			id = base + "-" + i;
		}
		return id;
	}

	public Path fileFor(String id) {
		if (!isValidId(id)) {
			throw new IllegalArgumentException("Invalid model id '" + id + "'");
		}
		return directory.resolve(id + ".json");
	}

	/** Validates and saves; saving under a built-in id creates an override. */
	public synchronized Entry save(String id, ModelDefinition model) throws IOException {
		model.requireValid();
		ModelJson.write(model, fileFor(id));
		final var entry = new Entry(id, model, BUILT_IN_IDS.contains(id) ? Source.MODIFIED_BUILT_IN : Source.USER);
		entries.put(id, entry);
		return entry;
	}

	/**
	 * Deletes a user model, or removes the override of a built-in.
	 *
	 * @return the entry now registered under the id (the restored built-in), if any
	 */
	public synchronized Optional<Entry> delete(String id) throws IOException {
		final var entry = entries.get(id);
		if (entry == null || entry.source() == Source.BUILT_IN) {
			throw new IllegalArgumentException("Built-in examples cannot be deleted");
		}
		Files.deleteIfExists(fileFor(id));
		if (entry.source() == Source.MODIFIED_BUILT_IN) {
			final var restored = new Entry(id, builtIn(id), Source.BUILT_IN);
			entries.put(id, restored);
			return Optional.of(restored);
		}
		entries.remove(id);
		return Optional.empty();
	}
}
