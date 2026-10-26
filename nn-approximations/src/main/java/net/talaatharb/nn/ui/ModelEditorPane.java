package net.talaatharb.nn.ui;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Spinner;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import net.talaatharb.nn.config.ActivationSpec;
import net.talaatharb.nn.config.ActivationType;
import net.talaatharb.nn.config.DatasetSpec;
import net.talaatharb.nn.config.Expression;
import net.talaatharb.nn.config.FunctionDataset;
import net.talaatharb.nn.config.InitializationSpec;
import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.config.NetworkSpec;
import net.talaatharb.nn.config.TableDataset;
import net.talaatharb.nn.config.TrainingSpec;

/**
 * Form + JSON editor for a {@link ModelDefinition}. Input and output layer sizes follow the dataset;
 * stored weights survive only when the network topology is left unchanged.
 */
public class ModelEditorPane extends BorderPane {

	private final TextField name = new TextField();
	private final TextArea description = new TextArea();
	private final ComboBox<ModelDefinition.Task> task = new ComboBox<>();
	private final VBox hiddenLayers = new VBox(6);
	private final ActivationEditor outputActivation = new ActivationEditor();
	private final Label outputSize = new Label();
	private final ComboBox<InitializationSpec.Scheme> scheme = new ComboBox<>();
	private final TextField initMin = small("-1");
	private final TextField initMax = small("1");
	private final TextField initSeed = small("");
	private final TextField rate = small("0.1");
	private final TextField target = small("0.001");
	private final TextField maxEpochs = small("5000");
	private final TextField shuffleSeed = small("");

	private final ToggleGroup datasetMode = new ToggleGroup();
	private final RadioButton functionMode = new RadioButton("Function of inputs (grid)");
	private final RadioButton tableMode = new RadioButton("Table of samples");
	private final VBox variables = new VBox(6);
	private final TextArea expressions = new TextArea();
	private final TextField tableInputNames = new TextField();
	private final TextField tableOutputNames = new TextField();
	private final TextArea tableRows = new TextArea();
	private final VBox functionPane = new VBox(8);
	private final VBox tablePane = new VBox(8);

	private final JsonCodeArea json = new JsonCodeArea();
	private final Label errors = new Label();
	private final TabPane tabs = new TabPane();

	private ModelDefinition original;

	public ModelEditorPane() {
		description.setPrefRowCount(2);
		description.setWrapText(true);
		task.getItems().setAll(ModelDefinition.Task.values());
		scheme.getItems().setAll(InitializationSpec.Scheme.values());
		scheme.valueProperty().addListener((observable, oldValue, value) -> {
			initMin.setDisable(value != InitializationSpec.Scheme.UNIFORM);
			initMax.setDisable(value != InitializationSpec.Scheme.UNIFORM);
		});
		initSeed.setPromptText("random");
		shuffleSeed.setPromptText("random");
		initSeed.setPrefColumnCount(8);
		shuffleSeed.setPrefColumnCount(8);
		expressions.setPrefRowCount(3);
		expressions.setPromptText("One expression per output, e.g. sin(pi*x)");
		expressions.textProperty().addListener((observable, oldValue, value) -> updateOutputSize());
		tableOutputNames.textProperty().addListener((observable, oldValue, value) -> updateOutputSize());
		tableRows.textProperty().addListener((observable, oldValue, value) -> updateOutputSize());
		datasetMode.selectedToggleProperty().addListener((observable, oldValue, value) -> {
			functionPane.setVisible(value == functionMode);
			functionPane.setManaged(value == functionMode);
			tablePane.setVisible(value == tableMode);
			tablePane.setManaged(value == tableMode);
			updateOutputSize();
		});
		functionMode.setToggleGroup(datasetMode);
		tableMode.setToggleGroup(datasetMode);
		tableRows.setPrefRowCount(10);
		tableRows.setStyle("-fx-font-family: monospace;");
		tableRows.setPromptText("0, 1 -> 1");
		errors.setStyle("-fx-text-fill: #b71c1c;");
		errors.setWrapText(true);

		tabs.getTabs().addAll(new Tab("Model", scroll(modelForm())), new Tab("Data", scroll(dataForm())),
				new Tab("JSON", jsonForm()));
		tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
		tabs.getSelectionModel().selectedItemProperty().addListener((observable, oldValue, value) -> {
			if (value != null && value.getText().equals("JSON")) {
				refreshJson();
			}
		});
		setCenter(tabs);
		setBottom(errors);
		BorderPane.setMargin(errors, new Insets(6, 0, 0, 0));
		setPrefSize(760, 620);
	}

	private static TextField small(String text) {
		final var field = new TextField(text);
		field.setPrefColumnCount(6);
		return field;
	}

	private static ScrollPane scroll(VBox content) {
		content.setPadding(new Insets(10));
		final var pane = new ScrollPane(content);
		pane.setFitToWidth(true);
		return pane;
	}

	private static Label heading(String text) {
		final var label = new Label(text);
		label.setStyle("-fx-font-weight: bold; -fx-padding: 8 0 0 0;");
		return label;
	}

	private VBox modelForm() {
		final var general = new GridPane();
		general.setHgap(8);
		general.setVgap(6);
		general.addRow(0, new Label("Name"), name);
		general.addRow(1, new Label("Description"), description);
		general.addRow(2, new Label("Task"), task);
		GridPane.setHgrow(name, Priority.ALWAYS);
		final var addLayer = new Button("Add hidden layer");
		addLayer.setOnAction(event -> addHiddenLayer(new NetworkSpec.LayerSpec(4, ActivationSpec.of(ActivationType.TANH))));
		final var output = new HBox(8, new Label("Output layer:"), outputSize, new Label("activation"), outputActivation);
		output.setAlignment(Pos.CENTER_LEFT);
		final var init = new HBox(8, new Label("Scheme"), scheme, new Label("min"), initMin, new Label("max"), initMax,
				new Label("seed"), initSeed);
		init.setAlignment(Pos.CENTER_LEFT);
		final var training = new HBox(8, new Label("Learning rate"), rate, new Label("Target MSE"), target,
				new Label("Max epochs"), maxEpochs, new Label("Shuffle seed"), shuffleSeed);
		training.setAlignment(Pos.CENTER_LEFT);
		return new VBox(6, general, heading("Hidden layers (input size comes from the dataset)"), hiddenLayers,
				addLayer, heading("Output"), output, heading("Weight initialization"), init,
				heading("Training defaults"), training);
	}

	private VBox dataForm() {
		final var addVariable = new Button("Add input");
		addVariable.setOnAction(event -> addVariable(new FunctionDataset.Variable(nextVariableName(), -1, 1, 11)));
		final var help = new Label("Operators: + - * / % ^ ( ). Constants: pi, e. Functions: "
				+ String.join(", ", Expression.functionNames())
				+ ". Logic functions treat values >= 0.5 as true and return 0/1.");
		help.setWrapText(true);
		help.setStyle("-fx-text-fill: #555;");
		functionPane.getChildren().setAll(new Label("Inputs: every combination of evenly spaced values is sampled"),
				variables, addVariable, new Label("Output expressions (one per line, one output neuron each)"),
				expressions, help);
		final var names = new GridPane();
		names.setHgap(8);
		names.setVgap(6);
		names.addRow(0, new Label("Input names"), tableInputNames);
		names.addRow(1, new Label("Output names"), tableOutputNames);
		GridPane.setHgrow(tableInputNames, Priority.ALWAYS);
		tableInputNames.setPromptText("a, b");
		tableOutputNames.setPromptText("y");
		tablePane.getChildren().setAll(names,
				new Label("Rows: inputs -> outputs, comma or space separated, one sample per line"), tableRows);
		VBox.setVgrow(tableRows, Priority.ALWAYS);
		return new VBox(8, new HBox(16, functionMode, tableMode), outputSize(), functionPane, tablePane);
	}

	private Label outputSize() {
		final var label = new Label();
		label.textProperty().bind(outputSize.textProperty().concat(" (output layer size follows the dataset)"));
		return label;
	}

	private VBox jsonForm() {
		final var refresh = new Button("Refresh from form");
		refresh.setOnAction(event -> refreshJson());
		final var apply = new Button("Apply JSON to form");
		apply.setOnAction(event -> {
			try {
				final var parsed = ModelJson.read(json.getText());
				final var problems = parsed.validate();
				if (!problems.isEmpty()) {
					throw new ModelDefinition.InvalidModelException(problems);
				}
				load(parsed);
				errors.setText("JSON applied.");
			} catch (IOException | IllegalArgumentException exception) {
				errors.setText(exception.getMessage());
			}
		});
		final var box = new VBox(8, new HBox(8, refresh, apply), json.pane());
		VBox.setVgrow(json.pane(), Priority.ALWAYS);
		box.setPadding(new Insets(10));
		return box;
	}

	private void refreshJson() {
		try {
			json.setJson(ModelJson.write(build()));
			errors.setText("");
		} catch (IllegalArgumentException exception) {
			errors.setText(exception.getMessage());
		}
	}

	public void load(ModelDefinition model) {
		original = model;
		name.setText(model.name());
		description.setText(model.description());
		task.setValue(model.task());
		hiddenLayers.getChildren().clear();
		final var layers = model.network().layers();
		for (int i = 0; i < layers.size() - 1; i++) {
			addHiddenLayer(layers.get(i));
		}
		outputActivation.set(layers.get(layers.size() - 1).activation());
		final var init = model.initialization();
		scheme.setValue(init.scheme());
		initMin.setText(Float.toString(init.min()));
		initMax.setText(Float.toString(init.max()));
		initSeed.setText(init.seed() == null ? "" : Long.toString(init.seed()));
		rate.setText(Float.toString(model.training().learningRate()));
		target.setText(Float.toString(model.training().targetLoss()));
		maxEpochs.setText(Integer.toString(model.training().maxEpochs()));
		shuffleSeed.setText(model.training().shuffleSeed() == null ? "" : Long.toString(model.training().shuffleSeed()));
		variables.getChildren().clear();
		expressions.clear();
		if (model.dataset() instanceof FunctionDataset function) {
			datasetMode.selectToggle(functionMode);
			function.variables().forEach(this::addVariable);
			expressions.setText(String.join("\n", function.outputs()));
			final var table = (TableDataset) toTable(function);
			fillTable(table);
		} else if (model.dataset() instanceof TableDataset table) {
			datasetMode.selectToggle(tableMode);
			fillTable(table);
			addVariable(new FunctionDataset.Variable("x", -1, 1, 21));
		}
		errors.setText("");
		updateOutputSize();
	}

	private static DatasetSpec toTable(FunctionDataset function) {
		if (function.sampleCount() > 400) {
			return new TableDataset(function.inputNames(), function.outputNames(), new float[0][], new float[0][]);
		}
		final var data = function.toTrainingData();
		return new TableDataset(function.inputNames(), List.of(), data.getInput(), data.getOutput());
	}

	private void fillTable(TableDataset table) {
		tableInputNames.setText(String.join(", ", table.inputNames()));
		tableOutputNames.setText(table.outputNames().stream().allMatch(Expression.IDENTIFIER.asMatchPredicate())
				? String.join(", ", table.outputNames()) : "");
		final var text = new StringBuilder();
		for (int r = 0; r < table.inputs().length; r++) {
			text.append(join(table.inputs()[r])).append(" -> ").append(join(table.outputs()[r])).append('\n');
		}
		tableRows.setText(text.toString());
	}

	private static String join(float[] values) {
		final var parts = new ArrayList<String>();
		for (float value : values) {
			parts.add(value == Math.rint(value) && Math.abs(value) < 1e7 ? Long.toString((long) value)
					: Float.toString(value));
		}
		return String.join(", ", parts);
	}

	public int hiddenLayerCount() {
		return hiddenLayers.getChildren().size();
	}

	void addHiddenLayer(NetworkSpec.LayerSpec spec) {
		final var neurons = new Spinner<Integer>(1, NetworkSpec.MAX_NEURONS_PER_LAYER, spec.neurons());
		neurons.setEditable(true);
		neurons.setPrefWidth(80);
		final var activation = new ActivationEditor();
		activation.set(spec.activation());
		final var title = new Label();
		final var up = new Button("↑");
		final var down = new Button("↓");
		final var remove = new Button("Remove");
		final var row = new HBox(8, title, new Label("neurons"), neurons, new Label("activation"), activation, up, down,
				remove);
		row.setAlignment(Pos.CENTER_LEFT);
		row.setUserData(new LayerRow(neurons, activation));
		up.setOnAction(event -> move(row, -1));
		down.setOnAction(event -> move(row, 1));
		remove.setOnAction(event -> {
			hiddenLayers.getChildren().remove(row);
			renumber();
		});
		hiddenLayers.getChildren().add(row);
		renumber();
	}

	private record LayerRow(Spinner<Integer> neurons, ActivationEditor activation) {
	}

	private void move(HBox row, int offset) {
		final var children = hiddenLayers.getChildren();
		final int index = children.indexOf(row);
		final int target = index + offset;
		if (target >= 0 && target < children.size()) {
			children.remove(index);
			children.add(target, row);
			renumber();
		}
	}

	private void renumber() {
		for (int i = 0; i < hiddenLayers.getChildren().size(); i++) {
			((Label) ((HBox) hiddenLayers.getChildren().get(i)).getChildren().get(0)).setText("Hidden " + (i + 1));
		}
	}

	private String nextVariableName() {
		for (int i = 1;; i++) {
			final String candidate = "x" + i;
			final boolean used = variables.getChildren().stream()
					.anyMatch(node -> ((VariableRow) node.getUserData()).name().getText().trim().equals(candidate));
			if (!used) {
				return candidate;
			}
		}
	}

	private record VariableRow(TextField name, TextField min, TextField max, TextField samples) {
	}

	private void addVariable(FunctionDataset.Variable variable) {
		final var fields = new VariableRow(small(variable.name()), small(Float.toString(variable.min())),
				small(Float.toString(variable.max())), small(Integer.toString(variable.samples())));
		final var remove = new Button("Remove");
		final var row = new HBox(8, new Label("name"), fields.name(), new Label("min"), fields.min(), new Label("max"),
				fields.max(), new Label("samples"), fields.samples(), remove);
		row.setAlignment(Pos.CENTER_LEFT);
		row.setUserData(fields);
		remove.setOnAction(event -> variables.getChildren().remove(row));
		variables.getChildren().add(row);
	}

	private void updateOutputSize() {
		final int count;
		if (datasetMode.getSelectedToggle() == tableMode) {
			final var first = tableRows.getText().lines().filter(line -> line.contains("->")).findFirst();
			count = first.map(line -> numbers(line.substring(line.indexOf("->") + 2)).length).orElse(0);
		} else {
			count = (int) expressions.getText().lines().filter(line -> !line.isBlank()).count();
		}
		outputSize.setText(count + (count == 1 ? " neuron" : " neurons"));
	}

	private static String[] numbers(String text) {
		final String trimmed = text.trim();
		return trimmed.isEmpty() ? new String[0] : trimmed.split("[,;\\s]+");
	}

	/**
	 * Converts the form into a validated model.
	 *
	 * @throws ModelDefinition.InvalidModelException listing every problem found
	 */
	public ModelDefinition build() {
		final var problems = new ArrayList<String>();
		final DatasetSpec dataset = datasetMode.getSelectedToggle() == tableMode ? buildTable(problems)
				: buildFunction(problems);
		final var layers = new ArrayList<NetworkSpec.LayerSpec>();
		for (var node : hiddenLayers.getChildren()) {
			final var row = (LayerRow) node.getUserData();
			commit(row.neurons());
			layers.add(new NetworkSpec.LayerSpec(row.neurons().getValue(), row.activation().get(problems)));
		}
		layers.add(new NetworkSpec.LayerSpec(dataset == null ? 1 : Math.max(1, dataset.outputSize()),
				outputActivation.get(problems)));
		final var network = new NetworkSpec(dataset == null ? 1 : dataset.inputSize(), layers);
		final var init = new InitializationSpec(scheme.getValue(), parseFloat(initMin, "initialization min", problems),
				parseFloat(initMax, "initialization max", problems), parseSeed(initSeed, "initialization seed", problems));
		final var training = new TrainingSpec(parseFloat(rate, "learning rate", problems),
				parseFloat(target, "target MSE", problems), parseInt(maxEpochs, "max epochs", problems),
				parseSeed(shuffleSeed, "shuffle seed", problems));
		if (!problems.isEmpty() || dataset == null) {
			throw new ModelDefinition.InvalidModelException(problems);
		}
		final float[][][] weights = original != null && original.weights() != null
				&& original.network().equals(network) ? original.weights() : null;
		final var model = new ModelDefinition(ModelDefinition.FEED_FORWARD, ModelDefinition.FORMAT_VERSION,
				name.getText().trim(), description.getText().trim(), task.getValue(), network, init, training, dataset,
				weights);
		return model.requireValid();
	}

	private static void commit(Spinner<Integer> spinner) {
		try {
			spinner.getValueFactory().setValue(Integer.parseInt(spinner.getEditor().getText().trim()));
		} catch (NumberFormatException exception) {
			spinner.getEditor().setText(Integer.toString(spinner.getValue()));
		}
	}

	private DatasetSpec buildFunction(List<String> problems) {
		final var list = new ArrayList<FunctionDataset.Variable>();
		for (var node : variables.getChildren()) {
			final var row = (VariableRow) node.getUserData();
			final String variable = row.name().getText().trim();
			list.add(new FunctionDataset.Variable(variable, parseFloat(row.min(), variable + " min", problems),
					parseFloat(row.max(), variable + " max", problems),
					parseInt(row.samples(), variable + " samples", problems)));
		}
		final var outputs = expressions.getText().lines().map(String::trim).filter(line -> !line.isEmpty()).toList();
		if (list.isEmpty()) {
			problems.add("Add at least one input");
			return null;
		}
		return new FunctionDataset(list, outputs);
	}

	private DatasetSpec buildTable(List<String> problems) {
		final var inputs = new ArrayList<float[]>();
		final var outputs = new ArrayList<float[]>();
		final var lines = tableRows.getText().lines().toList();
		for (int i = 0; i < lines.size(); i++) {
			final String line = lines.get(i).trim();
			if (line.isEmpty() || line.startsWith("#")) {
				continue;
			}
			final int arrow = line.indexOf("->");
			if (arrow < 0) {
				problems.add("Table line " + (i + 1) + ": expected 'inputs -> outputs'");
				continue;
			}
			try {
				inputs.add(parseRow(line.substring(0, arrow)));
				outputs.add(parseRow(line.substring(arrow + 2)));
			} catch (NumberFormatException exception) {
				problems.add("Table line " + (i + 1) + ": " + exception.getMessage());
			}
		}
		if (inputs.isEmpty()) {
			problems.add("Add at least one table row such as '0, 1 -> 1'");
			return null;
		}
		final var inputNames = names(tableInputNames.getText());
		final var outputNames = names(tableOutputNames.getText());
		return new TableDataset(inputNames, outputNames, inputs.toArray(float[][]::new),
				outputs.toArray(float[][]::new));
	}

	private static List<String> names(String text) {
		return text.isBlank() ? List.of() : List.of(text.trim().split("\\s*,\\s*"));
	}

	private static float[] parseRow(String text) {
		final String[] parts = numbers(text);
		final float[] values = new float[parts.length];
		for (int i = 0; i < parts.length; i++) {
			values[i] = Float.parseFloat(parts[i]);
		}
		return values;
	}

	private static float parseFloat(TextField field, String label, List<String> problems) {
		try {
			final float value = Float.parseFloat(field.getText().trim());
			if (Float.isFinite(value)) {
				return value;
			}
		} catch (NumberFormatException ignored) {
			// reported below
		}
		problems.add(label + ": '" + field.getText() + "' is not a finite number");
		return 0;
	}

	private static int parseInt(TextField field, String label, List<String> problems) {
		try {
			return Integer.parseInt(field.getText().trim());
		} catch (NumberFormatException exception) {
			problems.add(label + ": '" + field.getText() + "' is not a whole number");
			return 0;
		}
	}

	static Long parseSeed(TextField field, String label, List<String> problems) {
		final String text = field.getText().trim();
		if (text.isEmpty()) {
			return null;
		}
		try {
			return Long.parseLong(text);
		} catch (NumberFormatException exception) {
			problems.add(label + ": '" + text + "' is not a whole number (leave blank for random)");
			return null;
		}
	}

	public void showErrors(List<String> messages) {
		errors.setText(String.join("\n", messages));
	}

	/** Activation choice plus fields for its parameters. */
	static final class ActivationEditor extends HBox {
		private final ComboBox<ActivationType> type = new ComboBox<>();
		private final Map<String, TextField> parameters = new LinkedHashMap<>();
		private final HBox parameterBox = new HBox(6);

		ActivationEditor() {
			super(6);
			setAlignment(Pos.CENTER_LEFT);
			parameterBox.setAlignment(Pos.CENTER_LEFT);
			type.getItems().setAll(ActivationType.values());
			type.valueProperty().addListener((observable, oldValue, value) -> rebuild(Map.of()));
			getChildren().addAll(type, parameterBox);
		}

		void set(ActivationSpec spec) {
			final var resolved = ActivationType.fromId(spec.type()).orElse(ActivationType.LINEAR);
			type.setValue(resolved);
			rebuild(spec.params());
		}

		private void rebuild(Map<String, Float> values) {
			parameters.clear();
			parameterBox.getChildren().clear();
			final var value = type.getValue();
			if (value == null) {
				return;
			}
			for (String parameter : value.parameters()) {
				final var field = small(Float.toString(values.getOrDefault(parameter, value.defaultValue(parameter))));
				field.setPrefColumnCount(4);
				parameters.put(parameter, field);
				parameterBox.getChildren().addAll(new Label(parameter), field);
			}
			if (!value.trainable()) {
				final var warning = new Label("⚠ zero gradient");
				warning.setStyle("-fx-text-fill: #b26a00;");
				parameterBox.getChildren().add(warning);
			}
		}

		ActivationSpec get(List<String> problems) {
			final var value = type.getValue() == null ? ActivationType.LINEAR : type.getValue();
			final var params = new LinkedHashMap<String, Float>();
			parameters.forEach((parameter, field) -> {
				final float parsed = parseFloat(field, value.id() + " " + parameter, problems);
				if (parsed != value.defaultValue(parameter)) {
					params.put(parameter, parsed);
				}
			});
			return new ActivationSpec(value.id(), params);
		}
	}
}
