package net.talaatharb.nn.ui;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
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
import net.talaatharb.nn.config.HuggingFaceDataset;
import net.talaatharb.nn.config.HuggingFaceTableDataset;
import net.talaatharb.nn.config.InitializationSpec;
import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.config.NetworkSpec;
import net.talaatharb.nn.config.TableDataset;
import net.talaatharb.nn.config.TrainingSpec;
import net.talaatharb.nn.data.DatasetCache;
import net.talaatharb.nn.data.HuggingFaceClient;

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
	private final RadioButton imageMode = new RadioButton("Hugging Face images");
	private final RadioButton hfTableMode = new RadioButton("Hugging Face table");
	private final HuggingFaceTableForm hfTable = new HuggingFaceTableForm(this::hfTableChanged);
	private boolean hfTableHadClasses;
	private final VBox variables = new VBox(6);
	private final TextArea expressions = new TextArea();
	private final TextField tableInputNames = new TextField();
	private final TextField tableOutputNames = new TextField();
	private final TextArea tableRows = new TextArea();
	private final VBox functionPane = new VBox(8);
	private final VBox tablePane = new VBox(8);
	private final VBox imagePane = new VBox(8);
	private final TextField hfRepository = new TextField();
	private final ComboBox<String> hfConfig = editableCombo();
	private final ComboBox<String> hfSplit = editableCombo();
	private final ComboBox<String> hfTestSplit = editableCombo();
	private final TextField hfImageColumn = small("image");
	private final TextField hfLabelColumn = small("label");
	private final TextArea hfClasses = new TextArea();
	private final TextField hfSamples = small("2000");
	private final TextField hfTestSamples = small("500");
	private final TextField hfWidth = small("28");
	private final TextField hfHeight = small("28");
	private final CheckBox hfGrayscale = new CheckBox("Grayscale");
	private final Button hfLoadInfo = new Button("Load info");
	private final Label hfInfo = new Label();
	private List<HuggingFaceClient.ConfigInfo> hfConfigs = List.of();
	private boolean syncingTask;

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
			imagePane.setVisible(value == imageMode);
			imagePane.setManaged(value == imageMode);
			hfTable.setVisible(value == hfTableMode);
			hfTable.setManaged(value == hfTableMode);
			syncTask(value == imageMode, oldValue == imageMode);
			if (value == hfTableMode) {
				hfTableChanged();
			}
			updateOutputSize();
		});
		task.valueProperty().addListener((observable, oldValue, value) -> {
			if (!syncingTask && value == ModelDefinition.Task.IMAGE_CLASSIFICATION) {
				datasetMode.selectToggle(imageMode);
			} else if (!syncingTask && oldValue == ModelDefinition.Task.IMAGE_CLASSIFICATION
					&& datasetMode.getSelectedToggle() == imageMode) {
				datasetMode.selectToggle(functionMode);
			}
		});
		hfClasses.textProperty().addListener((observable, oldValue, value) -> updateOutputSize());
		functionMode.setToggleGroup(datasetMode);
		tableMode.setToggleGroup(datasetMode);
		imageMode.setToggleGroup(datasetMode);
		hfTableMode.setToggleGroup(datasetMode);
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

	private static ComboBox<String> editableCombo() {
		final var combo = new ComboBox<String>();
		combo.setEditable(true);
		combo.setPrefWidth(150);
		return combo;
	}

	HuggingFaceTableForm tableForm() {
		return hfTable;
	}

	/** A categorical target means classification; a numeric one defaults to regression (0/1 targets may classify). */
	private void hfTableChanged() {
		if (datasetMode.getSelectedToggle() != hfTableMode) {
			return;
		}
		final boolean hasClasses = !hfTable.classNames().isEmpty();
		syncingTask = true;
		try {
			if (hasClasses) {
				task.setValue(ModelDefinition.Task.CLASSIFICATION);
			} else if (hfTableHadClasses || task.getValue() != ModelDefinition.Task.CLASSIFICATION) {
				task.setValue(ModelDefinition.Task.REGRESSION);
			}
		} finally {
			syncingTask = false;
		}
		hfTableHadClasses = hasClasses;
		updateOutputSize();
	}

	/** Keeps the task and the dataset kind consistent: image datasets go with image classification only. */
	private void syncTask(boolean images, boolean wasImages) {
		syncingTask = true;
		try {
			if (images) {
				task.setValue(ModelDefinition.Task.IMAGE_CLASSIFICATION);
			} else if (wasImages && task.getValue() == ModelDefinition.Task.IMAGE_CLASSIFICATION) {
				task.setValue(ModelDefinition.Task.REGRESSION);
			}
		} finally {
			syncingTask = false;
		}
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
		return new VBox(8, new HBox(16, functionMode, tableMode, imageMode, hfTableMode), outputSize(), functionPane,
				tablePane, imageForm(), hfTable);
	}

	private VBox imageForm() {
		hfRepository.setPromptText("owner/name, e.g. ylecun/mnist");
		hfClasses.setPrefRowCount(2);
		hfClasses.setWrapText(true);
		hfClasses.setPromptText("Class names in label order, comma separated (Load info fills them in)");
		hfTestSplit.setPromptText("none");
		hfInfo.setWrapText(true);
		hfInfo.setStyle("-fx-text-fill: #555;");
		hfLoadInfo.setOnAction(event -> loadHuggingFaceInfo());
		hfConfig.valueProperty().addListener((observable, oldValue, value) -> showConfig(value));
		final var source = new GridPane();
		source.setHgap(8);
		source.setVgap(6);
		source.addRow(0, new Label("Repository"), new HBox(8, hfRepository, hfLoadInfo));
		HBox.setHgrow(hfRepository, Priority.ALWAYS);
		source.addRow(1, new Label("Config"), hfConfig);
		source.addRow(2, new Label("Training split"),
				new HBox(8, hfSplit, new Label("samples"), hfSamples));
		source.addRow(3, new Label("Test split"),
				new HBox(8, hfTestSplit, new Label("samples"), hfTestSamples));
		source.addRow(4, new Label("Columns"),
				new HBox(8, new Label("image"), hfImageColumn, new Label("label"), hfLabelColumn));
		source.addRow(5, new Label("Resize to"),
				new HBox(8, hfWidth, new Label("x"), hfHeight, hfGrayscale));
		source.addRow(6, new Label("Classes"), hfClasses);
		GridPane.setHgrow(hfClasses, Priority.ALWAYS);
		for (var node : source.getChildren()) {
			if (node instanceof HBox box) {
				box.setAlignment(Pos.CENTER_LEFT);
			}
		}
		final var help = new Label("Images are downloaded from the Hugging Face Parquet export into "
				+ DatasetCache.defaultDirectory() + " the first time the model is opened (set HF_TOKEN for gated"
				+ " datasets). Each pixel is one input scaled to 0..1 and each class is one output neuron, so keep"
				+ " images small (at most " + HuggingFaceDataset.MAX_IMAGE_SIDE + " px a side).");
		help.setWrapText(true);
		help.setStyle("-fx-text-fill: #555;");
		imagePane.getChildren().setAll(source, hfInfo, help);
		return imagePane;
	}

	private void loadHuggingFaceInfo() {
		final String repository = hfRepository.getText().trim();
		if (!HuggingFaceClient.REPOSITORY.matcher(repository).matches()) {
			hfInfo.setText("Enter a repository such as ylecun/mnist first.");
			return;
		}
		hfLoadInfo.setDisable(true);
		hfInfo.setText("Asking Hugging Face about " + repository + "…");
		final var client = HuggingFaceClient.fromEnvironment();
		final var info = new Task<List<HuggingFaceClient.ConfigInfo>>() {
			@Override
			protected List<HuggingFaceClient.ConfigInfo> call() throws Exception {
				return client.info(repository);
			}
		};
		info.setOnSucceeded(event -> {
			hfLoadInfo.setDisable(false);
			hfConfigs = info.getValue();
			final String previous = hfConfig.getValue();
			hfConfig.getItems().setAll(hfConfigs.stream().map(HuggingFaceClient.ConfigInfo::config).toList());
			final String chosen = hfConfig.getItems().contains(previous) ? previous
					: hfConfig.getItems().isEmpty() ? "default" : hfConfig.getItems().get(0);
			hfConfig.setValue(null);
			hfConfig.setValue(chosen);
			fillFromInfo(chosen, client, repository);
		});
		info.setOnFailed(event -> {
			hfLoadInfo.setDisable(false);
			hfInfo.setText("Could not load dataset info: " + info.getException().getMessage());
		});
		Thread.ofVirtual().start(info);
	}

	private Optional<HuggingFaceClient.ConfigInfo> configInfo(String config) {
		return hfConfigs.stream().filter(info -> info.config().equals(config)).findFirst();
	}

	/** Lists the splits of a config the user picked after Load info. */
	private void showConfig(String config) {
		configInfo(config).ifPresent(info -> {
			final String split = hfSplit.getValue();
			final String test = hfTestSplit.getValue();
			hfSplit.getItems().setAll(info.splits().keySet());
			hfTestSplit.getItems().setAll(info.splits().keySet());
			hfSplit.setValue(split);
			hfTestSplit.setValue(test);
		});
	}

	/** Fills columns, classes, splits and image size from the reported features of one config. */
	private void fillFromInfo(String config, HuggingFaceClient client, String repository) {
		final var found = configInfo(config);
		if (found.isEmpty()) {
			hfInfo.setText("No configs reported for " + repository + ".");
			return;
		}
		final var info = found.get();
		final var problems = new ArrayList<String>();
		info.firstImage().ifPresentOrElse(feature -> hfImageColumn.setText(feature.name()),
				() -> problems.add("no image column"));
		info.firstClassLabel().ifPresentOrElse(feature -> {
			hfLabelColumn.setText(feature.name());
			hfClasses.setText(String.join(", ", feature.classNames()));
		}, () -> problems.add("no class-label column (enter the label column and class names yourself)"));
		final var splits = info.splits().keySet();
		final String train = splits.contains("train") ? "train" : splits.stream().findFirst().orElse("train");
		hfSplit.setValue(train);
		final String test = splits.contains("test") ? "test"
				: splits.contains("validation") ? "validation" : null;
		hfTestSplit.setValue(test);
		final var sizes = new StringBuilder();
		info.splits().forEach((split, count) -> sizes.append(sizes.isEmpty() ? "" : ", ").append(split)
				.append(count >= 0 ? " " + count : ""));
		hfInfo.setText(info.config() + ": " + sizes + (problems.isEmpty() ? "" : " — " + String.join("; ", problems))
				+ ". Checking the image size…");
		final String column = hfImageColumn.getText().trim();
		final var size = new Task<Optional<int[]>>() {
			@Override
			protected Optional<int[]> call() throws Exception {
				return client.firstImageSize(repository, info.config(), train, column);
			}
		};
		final String summary = info.config() + ": " + sizes
				+ (problems.isEmpty() ? "" : " — " + String.join("; ", problems));
		size.setOnSucceeded(event -> size.getValue().ifPresentOrElse(dimensions -> {
			final int width = Math.min(dimensions[0], HuggingFaceDataset.MAX_IMAGE_SIDE);
			final int height = Math.min(dimensions[1], HuggingFaceDataset.MAX_IMAGE_SIDE);
			hfWidth.setText(Integer.toString(width));
			hfHeight.setText(Integer.toString(height));
			hfInfo.setText(summary + ". Images are " + dimensions[0] + "x" + dimensions[1]
					+ (width != dimensions[0] || height != dimensions[1] ? " (resized to " + width + "x" + height + ")"
							: "")
					+ ".");
		}, () -> hfInfo.setText(summary + ".")));
		size.setOnFailed(event -> hfInfo.setText(summary + "."));
		Thread.ofVirtual().start(size);
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
		} else if (model.dataset() instanceof HuggingFaceTableDataset table) {
			datasetMode.selectToggle(hfTableMode);
			hfTable.load(table);
			task.setValue(model.task());
			addVariable(new FunctionDataset.Variable("x", -1, 1, 21));
			tableInputNames.clear();
			tableOutputNames.clear();
			tableRows.clear();
		} else if (model.dataset() instanceof HuggingFaceDataset images) {
			datasetMode.selectToggle(imageMode);
			fillImages(images);
			addVariable(new FunctionDataset.Variable("x", -1, 1, 21));
			tableInputNames.clear();
			tableOutputNames.clear();
			tableRows.clear();
		}
		errors.setText("");
		updateOutputSize();
	}

	private void fillImages(HuggingFaceDataset images) {
		hfConfigs = List.of();
		hfRepository.setText(images.repository());
		hfConfig.getItems().setAll(images.config());
		hfConfig.setValue(images.config());
		hfSplit.getItems().clear();
		hfTestSplit.getItems().clear();
		hfSplit.setValue(images.split());
		hfTestSplit.setValue(images.testSplit());
		hfImageColumn.setText(images.imageColumn());
		hfLabelColumn.setText(images.labelColumn());
		hfClasses.setText(String.join(", ", images.classes()));
		hfSamples.setText(Integer.toString(images.samples()));
		hfTestSamples.setText(Integer.toString(images.testSamples()));
		hfWidth.setText(Integer.toString(images.image().width()));
		hfHeight.setText(Integer.toString(images.image().height()));
		hfGrayscale.setSelected(images.image().grayscale());
		hfInfo.setText("");
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
		} else if (datasetMode.getSelectedToggle() == imageMode) {
			count = classNames().size();
		} else if (datasetMode.getSelectedToggle() == hfTableMode) {
			count = hfTable.outputCount();
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
		final var mode = datasetMode.getSelectedToggle();
		final DatasetSpec dataset = mode == tableMode ? buildTable(problems)
				: mode == imageMode ? buildImages(problems)
						: mode == hfTableMode ? hfTable.build(problems) : buildFunction(problems);
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

	private List<String> classNames() {
		return Arrays.stream(hfClasses.getText().split("[,\\n]")).map(String::trim).filter(name -> !name.isEmpty())
				.toList();
	}

	private static String comboText(ComboBox<String> combo) {
		final String typed = combo.getEditor().getText();
		final String text = typed == null || typed.isBlank() ? combo.getValue() : typed;
		return text == null ? "" : text.trim();
	}

	private DatasetSpec buildImages(List<String> problems) {
		final String repository = hfRepository.getText().trim();
		if (repository.isEmpty()) {
			problems.add("Enter a Hugging Face repository such as ylecun/mnist");
			return null;
		}
		final String testSplit = comboText(hfTestSplit);
		return new HuggingFaceDataset(repository, comboText(hfConfig), comboText(hfSplit),
				hfImageColumn.getText().trim(), hfLabelColumn.getText().trim(), classNames(),
				parseInt(hfSamples, "training samples", problems), testSplit.isEmpty() ? null : testSplit,
				testSplit.isEmpty() ? 0 : parseInt(hfTestSamples, "test samples", problems),
				new HuggingFaceDataset.ImageSpec(parseInt(hfWidth, "image width", problems),
						parseInt(hfHeight, "image height", problems), hfGrayscale.isSelected()));
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
