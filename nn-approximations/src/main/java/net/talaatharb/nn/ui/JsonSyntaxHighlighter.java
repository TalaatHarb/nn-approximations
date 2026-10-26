package net.talaatharb.nn.ui;

import java.util.Collection;
import java.util.Collections;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.fxmisc.richtext.model.StyleSpans;
import org.fxmisc.richtext.model.StyleSpansBuilder;

/** Style classes for JSON text: {@code key}, {@code string}, {@code number}, {@code keyword} and {@code brace}. */
public final class JsonSyntaxHighlighter {

	private static final Pattern TOKENS = Pattern.compile("(?<key>\"(?:\\\\.|[^\\\\\"])*\")(?=\\s*:)"
			+ "|(?<string>\"(?:\\\\.|[^\\\\\"])*\")" + "|(?<keyword>\\b(?:true|false|null)\\b)"
			+ "|(?<number>-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)" + "|(?<brace>[{}\\[\\]])");

	private static final String[] STYLES = { "key", "string", "keyword", "number", "brace" };

	private JsonSyntaxHighlighter() {
	}

	public static StyleSpans<Collection<String>> computeHighlighting(String text) {
		final var spans = new StyleSpansBuilder<Collection<String>>();
		if (text == null || text.isEmpty()) {
			return spans.add(Collections.emptyList(), 0).create();
		}
		final Matcher matcher = TOKENS.matcher(text);
		int last = 0;
		while (matcher.find()) {
			if (matcher.start() > last) {
				spans.add(Collections.emptyList(), matcher.start() - last);
			}
			spans.add(Collections.singleton(style(matcher)), matcher.end() - matcher.start());
			last = matcher.end();
		}
		if (last < text.length() || last == 0) {
			spans.add(Collections.emptyList(), text.length() - last);
		}
		return spans.create();
	}

	private static String style(Matcher matcher) {
		for (String style : STYLES) {
			if (matcher.group(style) != null) {
				return style;
			}
		}
		throw new IllegalStateException("unmatched token");
	}
}
