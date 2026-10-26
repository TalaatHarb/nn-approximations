package net.talaatharb.nn.config;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;

import net.talaatharb.nn.functions.ActivationFunction;

/**
 * JSON form: {@code "tanh"} or, with parameters, {@code {"type": "parametricRelu", "alpha": 0.2}}.
 */
@JsonSerialize(using = ActivationSpec.Serializer.class)
@JsonDeserialize(using = ActivationSpec.Deserializer.class)
public record ActivationSpec(String type, Map<String, Float> params) {

	public ActivationSpec {
		params = params == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(params));
	}

	public static ActivationSpec of(ActivationType type) {
		return new ActivationSpec(type.id(), Map.of());
	}

	public ActivationType resolve() {
		return ActivationType.fromId(type).orElseThrow(() -> new IllegalArgumentException(
				"Unknown activation '" + type + "'"));
	}

	public ActivationFunction create() {
		return resolve().create(params);
	}

	void validate(String where, List<String> errors) {
		final var resolved = ActivationType.fromId(type);
		if (resolved.isEmpty()) {
			errors.add(where + ": unknown activation '" + type + "'; use one of "
					+ java.util.Arrays.stream(ActivationType.values()).map(ActivationType::id).toList());
			return;
		}
		final var activation = resolved.get();
		params.forEach((name, value) -> {
			if (!activation.parameters().contains(name)) {
				errors.add(where + ": " + type + " has no parameter '" + name + "'");
			} else if (value == null || !Float.isFinite(value)) {
				errors.add(where + ": parameter '" + name + "' must be a finite number");
			}
		});
		if (activation == ActivationType.PIECEWISE_LINEAR) {
			final float min = params.getOrDefault("xMin", activation.defaultValue("xMin"));
			final float max = params.getOrDefault("xMax", activation.defaultValue("xMax"));
			if (!(min < max)) {
				errors.add(where + ": piecewiseLinear requires xMin < xMax");
			}
		}
	}

	/** Parameters with defaults filled in, so explicit default values compare equal to omitted ones. */
	private Map<String, Float> effectiveParams() {
		final var resolved = ActivationType.fromId(type);
		if (resolved.isEmpty()) {
			return params;
		}
		final Map<String, Float> effective = new LinkedHashMap<>(params);
		for (final String name : resolved.get().parameters()) {
			effective.putIfAbsent(name, resolved.get().defaultValue(name));
		}
		return effective;
	}

	@Override
	public boolean equals(Object other) {
		return other instanceof ActivationSpec spec && java.util.Objects.equals(type, spec.type)
				&& effectiveParams().equals(spec.effectiveParams());
	}

	@Override
	public int hashCode() {
		return java.util.Objects.hash(type, effectiveParams());
	}

	@Override
	public String toString() {
		return params.isEmpty() ? type : type + params;
	}

	static final class Serializer extends StdSerializer<ActivationSpec> {
		Serializer() {
			super(ActivationSpec.class);
		}

		@Override
		public void serialize(ActivationSpec value, JsonGenerator generator, SerializerProvider provider)
				throws IOException {
			if (value.params().isEmpty()) {
				generator.writeString(value.type());
				return;
			}
			generator.writeStartObject();
			generator.writeStringField("type", value.type());
			for (var entry : value.params().entrySet()) {
				generator.writeNumberField(entry.getKey(), entry.getValue());
			}
			generator.writeEndObject();
		}
	}

	static final class Deserializer extends StdDeserializer<ActivationSpec> {
		Deserializer() {
			super(ActivationSpec.class);
		}

		@Override
		public ActivationSpec deserialize(JsonParser parser, DeserializationContext context) throws IOException {
			final JsonNode node = parser.readValueAsTree();
			if (node.isTextual()) {
				return new ActivationSpec(node.asText(), Map.of());
			}
			if (!node.isObject() || !node.path("type").isTextual()) {
				return context.reportInputMismatch(ActivationSpec.class,
						"Activation must be a name such as \"tanh\" or an object with a \"type\"");
			}
			final var params = new LinkedHashMap<String, Float>();
			final var fields = node.fields();
			while (fields.hasNext()) {
				final var field = fields.next();
				if (field.getKey().equals("type")) {
					continue;
				}
				if (!field.getValue().isNumber()) {
					return context.reportInputMismatch(ActivationSpec.class,
							"Activation parameter '" + field.getKey() + "' must be a number");
				}
				params.put(field.getKey(), field.getValue().floatValue());
			}
			return new ActivationSpec(node.get("type").asText(), params);
		}
	}
}
