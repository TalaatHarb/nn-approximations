package net.talaatharb.nn.ui;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import net.talaatharb.nn.config.ExampleCatalog;
import net.talaatharb.nn.config.ModelJson;

class JsonSyntaxHighlighterTest {

	@Test
	void classifiesKeysValuesAndPunctuation() {
		final String json = "{ \"a\\\"b\" : [-1.5e3, true, null], \"s\": \"x: y\" }";
		assertEquals(List.of("brace:{", "key:\"a\\\"b\"", "brace:[", "number:-1.5e3", "keyword:true", "keyword:null",
				"brace:]", "key:\"s\"", "string:\"x: y\"", "brace:}"), tokens(json));
	}

	@Test
	void spansCoverTheWholeText() {
		final String json = ModelJson.write(ExampleCatalog.builtIn("xor"));
		assertEquals(json.length(), JsonSyntaxHighlighter.computeHighlighting(json).length());
		assertEquals(0, JsonSyntaxHighlighter.computeHighlighting("").length());
		assertEquals(3, JsonSyntaxHighlighter.computeHighlighting("   ").length());
	}

	private static List<String> tokens(String text) {
		final var result = new ArrayList<String>();
		int position = 0;
		for (var span : JsonSyntaxHighlighter.computeHighlighting(text)) {
			if (!span.getStyle().isEmpty()) {
				result.add(span.getStyle().iterator().next() + ":"
						+ text.substring(position, position + span.getLength()));
			}
			position += span.getLength();
		}
		return result;
	}
}
