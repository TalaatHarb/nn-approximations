package net.talaatharb.nn.export;

import java.io.IOException;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import net.talaatharb.nn.config.ActivationSpec;
import net.talaatharb.nn.config.ActivationType;
import net.talaatharb.nn.config.DataScaling;
import net.talaatharb.nn.config.DatasetSpec;
import net.talaatharb.nn.config.ModelDefinition;

/**
 * Writes a trained model as a PMML 4.4 {@code NeuralNetwork} (regression over all outputs) for PMML consumers such
 * as JPMML, PMML4S/PyPMML, KNIME or SAS. PMML only defines identity, logistic, tanh, rectifier and threshold among the
 * activations used here; models with other activations should be exported as ONNX.
 */
public final class PmmlExporter {

	private static final String NAMESPACE = "http://www.dmg.org/PMML-4_4";

	private PmmlExporter() {
	}

	/** Empty when the model can be expressed in PMML, otherwise the reason it cannot. */
	public static Optional<String> unsupportedReason(ModelDefinition model) {
		final var layers = model.network().layers();
		for (int l = 0; l < layers.size(); l++) {
			if (pmmlActivation(layers.get(l).activation()) == null) {
				return Optional.of("PMML has no " + layers.get(l).activation().resolve() + " activation (layer "
						+ (l + 1) + "); use ONNX instead");
			}
		}
		final var names = new HashSet<String>();
		for (String name : fieldNames(model)) {
			if (!names.add(name)) {
				return Optional.of("PMML needs distinct input and output names, but '" + name + "' repeats");
			}
		}
		return Optional.empty();
	}

	public static void write(ModelDefinition model, float[][][] weights, Path path) throws IOException {
		write(model, weights, model.dataset().scaling(), path);
	}

	public static void write(ModelDefinition model, float[][][] weights, DataScaling scaling, Path path)
			throws IOException {
		ExportFiles.writeAtomically(path, toXml(model, weights, scaling).getBytes(StandardCharsets.UTF_8));
	}

	/** Uses the dataset's {@link DatasetSpec#scaling() scaling}, so consumers pass raw column values. */
	public static String toXml(ModelDefinition model, float[][][] weights) {
		return toXml(model, weights, model.dataset().scaling());
	}

	/**
	 * Inputs and outputs whose scaling is not the identity get a {@code NormContinuous} transformation: PMML
	 * consumers feed raw values and receive predictions in target units.
	 */
	public static String toXml(ModelDefinition model, float[][][] weights, DataScaling scaling) {
		ExportFiles.requireCompatible(model, weights);
		ExportFiles.requireCompatible(model, scaling);
		unsupportedReason(model).ifPresent(reason -> {
			throw new IllegalArgumentException(reason);
		});
		final var text = new StringWriter();
		try {
			final XMLStreamWriter xml = XMLOutputFactory.newFactory().createXMLStreamWriter(text);
			xml.writeStartDocument("UTF-8", "1.0");
			xml.writeCharacters("\n");
			xml.writeStartElement("PMML");
			xml.writeDefaultNamespace(NAMESPACE);
			xml.writeAttribute("version", "4.4");
			xml.writeStartElement("Header");
			xml.writeAttribute("description", model.description().isBlank() ? model.name() : model.description());
			xml.writeEmptyElement("Application");
			xml.writeAttribute("name", "nn-approximations");
			xml.writeEndElement();

			final var inputs = model.dataset().inputNames();
			final var outputs = model.dataset().outputNames();
			xml.writeStartElement("DataDictionary");
			xml.writeAttribute("numberOfFields", Integer.toString(inputs.size() + outputs.size()));
			for (String name : fieldNames(model)) {
				xml.writeEmptyElement("DataField");
				xml.writeAttribute("name", name);
				xml.writeAttribute("optype", "continuous");
				xml.writeAttribute("dataType", "double");
			}
			xml.writeEndElement();

			final var layers = model.network().layers();
			xml.writeStartElement("NeuralNetwork");
			xml.writeAttribute("modelName", model.name());
			xml.writeAttribute("functionName", "regression");
			xml.writeAttribute("activationFunction", "identity");
			xml.writeAttribute("numberOfLayers", Integer.toString(layers.size()));
			xml.writeStartElement("MiningSchema");
			for (String name : inputs) {
				xml.writeEmptyElement("MiningField");
				xml.writeAttribute("name", name);
			}
			for (String name : outputs) {
				xml.writeEmptyElement("MiningField");
				xml.writeAttribute("name", name);
				xml.writeAttribute("usageType", "target");
			}
			xml.writeEndElement();

			xml.writeStartElement("NeuralInputs");
			xml.writeAttribute("numberOfInputs", Integer.toString(inputs.size()));
			for (int i = 0; i < inputs.size(); i++) {
				xml.writeStartElement("NeuralInput");
				xml.writeAttribute("id", "0," + i);
				field(xml, inputs.get(i), scaling.inputOffset()[i], scaling.inputScale()[i]);
				xml.writeEndElement();
			}
			xml.writeEndElement();

			for (int l = 0; l < layers.size(); l++) {
				final var activation = layers.get(l).activation();
				xml.writeStartElement("NeuralLayer");
				xml.writeAttribute("numberOfNeurons", Integer.toString(weights[l].length));
				xml.writeAttribute("activationFunction", pmmlActivation(activation));
				if (activation.resolve() == ActivationType.STEP) {
					xml.writeAttribute("threshold", number(activation.params().getOrDefault("threshold",
							ActivationType.STEP.defaultValue("threshold"))));
				}
				for (int n = 0; n < weights[l].length; n++) {
					xml.writeStartElement("Neuron");
					xml.writeAttribute("id", (l + 1) + "," + n);
					xml.writeAttribute("bias", number(weights[l][n][0]));
					for (int i = 1; i < weights[l][n].length; i++) {
						xml.writeEmptyElement("Con");
						xml.writeAttribute("from", l + "," + (i - 1));
						xml.writeAttribute("weight", number(weights[l][n][i]));
					}
					xml.writeEndElement();
				}
				xml.writeEndElement();
			}

			xml.writeStartElement("NeuralOutputs");
			xml.writeAttribute("numberOfOutputs", Integer.toString(outputs.size()));
			for (int o = 0; o < outputs.size(); o++) {
				xml.writeStartElement("NeuralOutput");
				xml.writeAttribute("outputNeuron", layers.size() + "," + o);
				field(xml, outputs.get(o), scaling.outputOffset()[o], scaling.outputScale()[o]);
				xml.writeEndElement();
			}
			xml.writeEndElement();
			xml.writeEndElement();
			xml.writeEndElement();
			xml.writeEndDocument();
			xml.close();
		} catch (XMLStreamException exception) {
			throw new IllegalStateException("Could not write PMML", exception);
		}
		return text.toString();
	}

	private static List<String> fieldNames(ModelDefinition model) {
		return java.util.stream.Stream
				.concat(model.dataset().inputNames().stream(), model.dataset().outputNames().stream()).toList();
	}

	private static String pmmlActivation(ActivationSpec activation) {
		return switch (activation.resolve()) {
			case LINEAR -> "identity";
			case SIGMOID -> "logistic";
			case TANH -> "tanh";
			case RELU -> "rectifier";
			case STEP -> "threshold";
			default -> null;
		};
	}

	/** The field as is, or normalized as (value - offset) / scale through two LinearNorm points. */
	private static void field(XMLStreamWriter xml, String field, float offset, float scale)
			throws XMLStreamException {
		xml.writeStartElement("DerivedField");
		xml.writeAttribute("optype", "continuous");
		xml.writeAttribute("dataType", "double");
		if (offset == 0 && scale == 1) {
			xml.writeEmptyElement("FieldRef");
			xml.writeAttribute("field", field);
		} else {
			xml.writeStartElement("NormContinuous");
			xml.writeAttribute("field", field);
			xml.writeEmptyElement("LinearNorm");
			xml.writeAttribute("orig", number(offset));
			xml.writeAttribute("norm", "0");
			xml.writeEmptyElement("LinearNorm");
			xml.writeAttribute("orig", Double.toString((double) offset + scale));
			xml.writeAttribute("norm", "1");
			xml.writeEndElement();
		}
		xml.writeEndElement();
	}

	/** The exact value of the float, so double-precision consumers use the trained weights unchanged. */
	private static String number(float value) {
		return Double.toString(value);
	}
}
