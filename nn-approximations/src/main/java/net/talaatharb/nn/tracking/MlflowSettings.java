package net.talaatharb.nn.tracking;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

/**
 * Where training runs are logged. Stored in {@code ~/.nn-approximations/mlflow.json} (override the file with the
 * {@code nn.mlflow.settings} system property). Credentials are never stored here; they come from the standard
 * {@code MLFLOW_TRACKING_TOKEN} or {@code MLFLOW_TRACKING_USERNAME}/{@code MLFLOW_TRACKING_PASSWORD} variables.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MlflowSettings(boolean enabled, String trackingUri, String experimentName) {

	public static final String DEFAULT_TRACKING_URI = "https://mlflow.talaatharb.net";
	public static final String DEFAULT_EXPERIMENT = "nn-approximations";

	private static final ObjectMapper MAPPER = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

	public MlflowSettings {
		trackingUri = trackingUri == null || trackingUri.isBlank() ? defaultUri() : trackingUri.trim();
		while (trackingUri.endsWith("/")) {
			trackingUri = trackingUri.substring(0, trackingUri.length() - 1);
		}
		experimentName = experimentName == null || experimentName.isBlank() ? defaultExperiment()
				: experimentName.trim();
	}

	/** Defaults: system properties {@code nn.mlflow.*}, then the standard MLFLOW_* variables, then built-ins. */
	public static MlflowSettings defaults() {
		return new MlflowSettings(Boolean.getBoolean("nn.mlflow.enabled"), null, null);
	}

	public static Path settingsFile() {
		final String override = System.getProperty("nn.mlflow.settings");
		return override != null && !override.isBlank() ? Path.of(override)
				: Path.of(System.getProperty("user.home"), ".nn-approximations", "mlflow.json");
	}

	/** Saved settings, or {@link #defaults()} when the file is missing or unreadable. */
	public static MlflowSettings load() {
		final Path file = settingsFile();
		if (!Files.isRegularFile(file)) {
			return defaults();
		}
		try {
			return MAPPER.readValue(file.toFile(), MlflowSettings.class);
		} catch (IOException exception) {
			return defaults();
		}
	}

	public void save() throws IOException {
		final Path file = settingsFile().toAbsolutePath();
		Files.createDirectories(file.getParent());
		MAPPER.writeValue(file.toFile(), this);
	}

	public List<String> validate() {
		final var errors = new ArrayList<String>();
		try {
			final URI uri = URI.create(trackingUri);
			final String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
			if (!scheme.equals("http") && !scheme.equals("https") || uri.getHost() == null) {
				errors.add("Tracking URI must be an http(s) URL such as " + DEFAULT_TRACKING_URI);
			}
		} catch (IllegalArgumentException exception) {
			errors.add("Tracking URI is not a valid URL: " + exception.getMessage());
		}
		return errors;
	}

	public URI uri() {
		return URI.create(trackingUri);
	}

	public MlflowSettings withEnabled(boolean value) {
		return new MlflowSettings(value, trackingUri, experimentName);
	}

	private static String defaultUri() {
		return firstNonBlank(System.getProperty("nn.mlflow.uri"), System.getenv("MLFLOW_TRACKING_URI"),
				DEFAULT_TRACKING_URI);
	}

	private static String defaultExperiment() {
		return firstNonBlank(System.getProperty("nn.mlflow.experiment"), System.getenv("MLFLOW_EXPERIMENT_NAME"),
				DEFAULT_EXPERIMENT);
	}

	private static String firstNonBlank(String... values) {
		for (String value : values) {
			if (value != null && !value.isBlank()) {
				return value.trim();
			}
		}
		return null;
	}
}
