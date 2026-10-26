package net.talaatharb.nn.ui;

import java.util.concurrent.CompletableFuture;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
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
		final var check = new Button("Test connection");
		final var result = new Label();
		result.setWrapText(true);
		result.setMaxWidth(460);
		check.setOnAction(event -> {
			final var candidate = read(enabled, uri, experiment);
			final var errors = candidate.validate();
			if (!errors.isEmpty()) {
				result.setText(String.join("\n", errors));
				return;
			}
			check.setDisable(true);
			result.setText("Connecting to " + candidate.trackingUri() + "...");
			CompletableFuture.supplyAsync(() -> {
				try {
					return "Connected: MLflow " + new MlflowClient(candidate.uri()).version();
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
		grid.addRow(3, new Label(""), new HBox(8, check));
		grid.add(result, 1, 4);
		grid.add(new Label("Credentials, if the server needs them, are read from MLFLOW_TRACKING_TOKEN or\n"
				+ "MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD and are never saved."), 0, 5, 2, 1);
		getDialogPane().setContent(grid);
		getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
		getDialogPane().lookupButton(ButtonType.OK).addEventFilter(javafx.event.ActionEvent.ACTION, event -> {
			final var errors = read(enabled, uri, experiment).validate();
			if (!errors.isEmpty()) {
				result.setText(String.join("\n", errors));
				event.consume();
			}
		});
		setResultConverter(button -> button == ButtonType.OK ? read(enabled, uri, experiment) : null);
	}

	private static MlflowSettings read(CheckBox enabled, TextField uri, TextField experiment) {
		return new MlflowSettings(enabled.isSelected(), uri.getText(), experiment.getText());
	}
}
