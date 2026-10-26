package net.talaatharb.nn.ui;

import java.util.concurrent.CompletableFuture;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.PasswordField;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.stage.Window;
import net.talaatharb.nn.tracking.MlflowClient;
import net.talaatharb.nn.tracking.MlflowSettings;

/** Edits where training runs are logged and checks that the server is reachable. */
final class MlflowSettingsDialog extends Dialog<MlflowSettings> {

	MlflowSettingsDialog(Window owner, MlflowSettings current) {
		initOwner(owner);
		setTitle("MLflow tracking");
		setHeaderText("Log each training session as an MLflow run (loss per epoch, settings, trained model)");
		final var enabled = new CheckBox("Log training runs");
		enabled.setSelected(current.enabled());
		final var uri = new TextField(current.trackingUri());
		uri.setPrefColumnCount(32);
		uri.setPromptText(MlflowSettings.DEFAULT_TRACKING_URI);
		final var experiment = new TextField(current.experimentName());
		experiment.setPromptText(MlflowSettings.DEFAULT_EXPERIMENT);
		final var username = new TextField(current.username() == null ? "" : current.username());
		username.setPromptText("optional");
		final var password = new PasswordField();
		password.setText(current.password() == null ? "" : current.password());
		password.disableProperty().bind(username.textProperty().isEmpty());
		final var savePassword = new CheckBox("Save password in " + MlflowSettings.settingsFile());
		savePassword.setSelected(current.savePassword());
		savePassword.disableProperty().bind(password.disableProperty());
		final Fields fields = new Fields(enabled, uri, experiment, username, password, savePassword);
		final var check = new Button("Test connection");
		final var result = new Label();
		result.setWrapText(true);
		result.setMaxWidth(460);
		check.setOnAction(event -> {
			final var candidate = fields.read();
			final var errors = candidate.validate();
			if (!errors.isEmpty()) {
				result.setText(String.join("\n", errors));
				return;
			}
			check.setDisable(true);
			result.setText("Connecting to " + candidate.trackingUri() + "...");
			CompletableFuture.supplyAsync(() -> {
				try {
					final var client = MlflowClient.connect(candidate);
					final String version = client.version();
					client.checkAccess();
					return "Connected: MLflow " + version + (candidate.hasCredentials()
							? ", signed in as " + candidate.username() : "");
				} catch (Exception exception) {
					return "Not reachable: " + exception.getMessage();
				}
			}).thenAccept(message -> Platform.runLater(() -> {
				result.setText(message);
				check.setDisable(false);
			}));
		});

		final var grid = new GridPane();
		grid.setHgap(8);
		grid.setVgap(8);
		grid.setPadding(new Insets(12));
		grid.addRow(0, new Label(""), enabled);
		grid.addRow(1, new Label("Tracking URI"), uri);
		grid.addRow(2, new Label("Experiment"), experiment);
		grid.addRow(3, new Label("Username"), username);
		grid.addRow(4, new Label("Password"), password);
		grid.addRow(5, new Label(""), savePassword);
		grid.addRow(6, new Label(""), new HBox(8, check));
		grid.add(result, 1, 7);
		final var note = new Label("Username and password are sent with HTTP basic authentication. A saved password is "
				+ "stored as plain text; otherwise it is kept until the application closes. Without a username, "
				+ "MLFLOW_TRACKING_TOKEN or MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD are used.");
		note.setStyle("-fx-text-fill: #555;");
		note.setWrapText(true);
		note.setMaxWidth(520);
		grid.add(note, 0, 8, 2, 1);
		getDialogPane().setContent(grid);
		getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
		getDialogPane().lookupButton(ButtonType.OK).addEventFilter(javafx.event.ActionEvent.ACTION, event -> {
			final var errors = fields.read().validate();
			if (!errors.isEmpty()) {
				result.setText(String.join("\n", errors));
				event.consume();
			}
		});
		setResultConverter(button -> button == ButtonType.OK ? fields.read() : null);
	}

	private record Fields(CheckBox enabled, TextField uri, TextField experiment, TextField username,
			PasswordField password, CheckBox savePassword) {
		MlflowSettings read() {
			final boolean signIn = !username.getText().isBlank();
			return new MlflowSettings(enabled.isSelected(), uri.getText(), experiment.getText(), username.getText(),
					signIn ? password.getText() : null, signIn && savePassword.isSelected());
		}
	}
}
