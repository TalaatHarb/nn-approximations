package net.talaatharb.nn.tracking;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.AclEntry;
import java.nio.file.attribute.AclEntryPermission;
import java.nio.file.attribute.AclEntryType;
import java.nio.file.attribute.AclFileAttributeView;
import java.nio.file.attribute.PosixFileAttributeView;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

/**
 * Where training runs are logged. Stored in {@code ~/.nn-approximations/mlflow.json} (override the file with the
 * {@code nn.mlflow.settings} system property).
 * <p>
 * A {@code username} enables HTTP basic authentication with {@code password}. The password is written to the file
 * (in plain text, readable only by the owner where the file system allows it) only when {@code savePassword} is set;
 * otherwise it lives for the session. Without a username the standard {@code MLFLOW_TRACKING_TOKEN} or
 * {@code MLFLOW_TRACKING_USERNAME}/{@code MLFLOW_TRACKING_PASSWORD} variables are used.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MlflowSettings(boolean enabled, String trackingUri, String experimentName, String username,
		String password, boolean savePassword) {

	public static final String DEFAULT_TRACKING_URI = "https://mlflow.talaatharb.net";
	public static final String DEFAULT_EXPERIMENT = "nn-approximations";

	private static final ObjectMapper MAPPER = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

	@JsonCreator
	public MlflowSettings {
		username = username == null || username.isBlank() ? null : username.trim();
		password = password == null || password.isEmpty() ? null : password;
		trackingUri = trackingUri == null || trackingUri.isBlank() ? defaultUri() : trackingUri.trim();
		while (trackingUri.endsWith("/")) {
			trackingUri = trackingUri.substring(0, trackingUri.length() - 1);
		}
		experimentName = experimentName == null || experimentName.isBlank() ? defaultExperiment()
				: experimentName.trim();
	}

	public MlflowSettings(boolean enabled, String trackingUri, String experimentName) {
		this(enabled, trackingUri, experimentName, null, null, false);
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

	/** Writes the settings; the password only when {@link #savePassword()} is set. */
	public void save() throws IOException {
		final Path file = settingsFile().toAbsolutePath();
		Files.createDirectories(file.getParent());
		final var stored = savePassword ? this : withPassword(null);
		if (!Files.exists(file)) {
			Files.createFile(file);
		}
		restrictToOwner(file);
		Files.write(file, MAPPER.writeValueAsBytes(stored));
	}

	/** Basic authentication is used when a username is set. */
	public boolean hasCredentials() {
		return username != null;
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
		return new MlflowSettings(value, trackingUri, experimentName, username, password, savePassword);
	}

	public MlflowSettings withPassword(String value) {
		return new MlflowSettings(enabled, trackingUri, experimentName, username, value, savePassword);
	}

	@Override
	public String toString() {
		return "MlflowSettings[enabled=" + enabled + ", trackingUri=" + trackingUri + ", experimentName="
				+ experimentName + ", username=" + username + ", password=" + (password == null ? null : "****")
				+ ", savePassword=" + savePassword + "]";
	}

	/** Best effort: the file may hold a password. */
	private static void restrictToOwner(Path file) {
		try {
			final var posix = Files.getFileAttributeView(file, PosixFileAttributeView.class);
			if (posix != null) {
				posix.setPermissions(PosixFilePermissions.fromString("rw-------"));
				return;
			}
			final var acl = Files.getFileAttributeView(file, AclFileAttributeView.class);
			if (acl != null) {
				acl.setAcl(List.of(AclEntry.newBuilder().setType(AclEntryType.ALLOW).setPrincipal(acl.getOwner())
						.setPermissions(EnumSet.allOf(AclEntryPermission.class)).build()));
			}
		} catch (IOException | UnsupportedOperationException | SecurityException exception) {
			// keep the default permissions
		}
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
