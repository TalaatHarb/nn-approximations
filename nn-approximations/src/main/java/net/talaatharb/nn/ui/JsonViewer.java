package net.talaatharb.nn.ui;

import java.util.Locale;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.stage.Stage;
import javafx.stage.Window;
import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;

/** Read-only, highlighted JSON of a model, optionally with a snapshot of the network's current weights. */
final class JsonViewer extends Stage {

	private final JsonCodeArea area = new JsonCodeArea();
	private final Label info = new Label();

	JsonViewer(Window owner, ModelDefinition model, float[][][] currentWeights) {
		initOwner(owner);
		setTitle(model.name() + " - JSON");
		area.setEditable(false);
		final var withWeights = new CheckBox("Include current weights");
		withWeights.setSelected(true);
		withWeights.selectedProperty().addListener((observable, oldValue, value) -> show(value
				? model.withWeights(currentWeights) : model.withWeights(null)));
		final var copy = new Button("Copy");
		copy.setOnAction(event -> {
			final var content = new ClipboardContent();
			content.putString(area.getText());
			Clipboard.getSystemClipboard().setContent(content);
			info.setText("Copied " + area.getLength() + " characters");
		});
		final var find = new TextField();
		find.setPromptText("Find (Enter for next)");
		find.setPrefColumnCount(16);
		find.setOnAction(event -> findNext(find.getText()));
		final var wrap = new CheckBox("Wrap lines");
		wrap.selectedProperty().bindBidirectional(area.wrapTextProperty());
		wrap.setSelected(true);
		final var spacer = new Region();
		HBox.setHgrow(spacer, Priority.ALWAYS);
		final var toolbar = new HBox(8, withWeights, wrap, copy, find, spacer, info);
		toolbar.setAlignment(Pos.CENTER_LEFT);
		toolbar.setPadding(new Insets(8));
		final var root = new BorderPane(area.pane());
		root.setTop(toolbar);
		setScene(new Scene(root, 640, 720));
		show(model.withWeights(currentWeights));
	}

	private void show(ModelDefinition model) {
		area.setJson(ModelJson.write(model));
		info.setText(area.getParagraphs().size() + " lines");
	}

	private void findNext(String text) {
		if (text.isEmpty()) {
			return;
		}
		final String haystack = area.getText().toLowerCase(Locale.ROOT);
		final String needle = text.toLowerCase(Locale.ROOT);
		int index = haystack.indexOf(needle, area.getSelection().getEnd());
		if (index < 0) {
			index = haystack.indexOf(needle);
		}
		if (index < 0) {
			info.setText("Not found: " + text);
			return;
		}
		area.selectRange(index, index + needle.length());
		area.requestFollowCaret();
		info.setText("Line " + (area.getCurrentParagraph() + 1));
	}
}
