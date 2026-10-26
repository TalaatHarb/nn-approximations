package net.talaatharb.nn.ui;

import javafx.event.ActionEvent;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.stage.Window;
import net.talaatharb.nn.config.ModelDefinition;

/** Modal editor; OK stays disabled-in-effect until the form builds a valid model. */
public class ModelEditorDialog extends Dialog<ModelDefinition> {

	private final ModelEditorPane editor = new ModelEditorPane();
	private ModelDefinition result;

	public ModelEditorDialog(Window owner, String title, ModelDefinition initial) {
		initOwner(owner);
		setTitle(title);
		setResizable(true);
		editor.load(initial);
		getDialogPane().setContent(editor);
		getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
		getDialogPane().lookupButton(ButtonType.OK).addEventFilter(ActionEvent.ACTION, event -> {
			try {
				result = editor.build();
			} catch (ModelDefinition.InvalidModelException exception) {
				editor.showErrors(exception.errors());
				event.consume();
			}
		});
		setResultConverter(button -> button == ButtonType.OK ? result : null);
	}

	public ModelEditorPane editor() {
		return editor;
	}
}
