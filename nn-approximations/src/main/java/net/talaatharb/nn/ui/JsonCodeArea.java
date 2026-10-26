package net.talaatharb.nn.ui;

import java.time.Duration;

import org.fxmisc.flowless.VirtualizedScrollPane;
import org.fxmisc.richtext.CodeArea;
import org.fxmisc.richtext.LineNumberFactory;

/** JSON editor/viewer with line numbers and syntax highlighting; use {@link #pane()} in layouts. */
final class JsonCodeArea extends CodeArea {

	private final VirtualizedScrollPane<JsonCodeArea> pane = new VirtualizedScrollPane<>(this);

	JsonCodeArea() {
		getStyleClass().add("json-area");
		getStylesheets().add(JsonCodeArea.class.getResource("json-area.css").toExternalForm());
		setParagraphGraphicFactory(LineNumberFactory.get(this));
		setWrapText(true);
		// Typing is re-highlighted once it pauses; programmatic replacements are highlighted immediately.
		multiPlainChanges().successionEnds(Duration.ofMillis(150)).subscribe(ignore -> highlight());
	}

	VirtualizedScrollPane<JsonCodeArea> pane() {
		return pane;
	}

	void setJson(String json) {
		replaceText(json);
		highlight();
		moveTo(0);
		requestFollowCaret();
	}

	private void highlight() {
		if (getLength() > 0) {
			setStyleSpans(0, JsonSyntaxHighlighter.computeHighlighting(getText()));
		}
	}
}
