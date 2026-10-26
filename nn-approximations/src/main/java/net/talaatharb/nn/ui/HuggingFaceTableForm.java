package net.talaatharb.nn.ui;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

import javafx.concurrent.Task;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import net.talaatharb.nn.config.DataScaling;
import net.talaatharb.nn.config.HuggingFaceTableDataset;
import net.talaatharb.nn.data.DatasetCache;
import net.talaatharb.nn.data.HuggingFaceClient;

/**
 * Data form of a tabular Hugging Face dataset: pick the input columns and the target, optionally list
 * the target's classes, and choose how rows are split and scaled. {@code Load info} asks the datasets
 * server for the columns and for statistics that reveal categorical targets.
 */
final class HuggingFaceTableForm extends VBox {

	private static final Pattern ID_LIKE = Pattern.compile("(?i)id|index|idx|row_?id|unnamed.*|__index_level_\\d+__");
	private static final Pattern TARGET_LIKE = Pattern.compile("(?i)label|labels|target|class|species|y");

	final TextField repository = new TextField();
	final Button loadInfo = new Button("Load info");
	final ComboBox<String> config = combo();
	final ComboBox<String> split = combo();
	final ComboBox<String> testSplit = combo();
	final TextField samples = field("1000");
	final TextField testSamples = field("200");
	final TextField seed = field("0");
	final FlowPane columns = new FlowPane(10, 4);
	final ComboBox<String> target = combo();
	final TextArea classes = new TextArea();
	final ComboBox<DataScaling.Mode> inputScaling = new ComboBox<>();
	final ComboBox<DataScaling.Mode> targetScaling = new ComboBox<>();
	final Label info = new Label();

	private final Runnable onChange;
	private List<HuggingFaceClient.ConfigInfo> configs = List.of();
	/** Statistics of the training split per column (empty when unknown). */
	private Map<String, HuggingFaceClient.ColumnStatistics> statistics = Map.of();
	private boolean filling;

	HuggingFaceTableForm(Runnable onChange) {
		super(8);
		this.onChange = onChange;
		repository.setPromptText("owner/name, e.g. scikit-learn/iris");
		testSplit.setPromptText("hold out rows");
		classes.setPrefRowCount(2);
		classes.setWrapText(true);
		classes.setPromptText("Class names for a categorical target, comma separated (empty = numeric target)");
		inputScaling.getItems().setAll(DataScaling.Mode.values());
		targetScaling.getItems().setAll(DataScaling.Mode.values());
		inputScaling.setValue(DataScaling.Mode.STANDARDIZE);
		targetScaling.setValue(DataScaling.Mode.STANDARDIZE);
		info.setWrapText(true);
		info.setStyle("-fx-text-fill: #555;");
		loadInfo.setOnAction(event -> loadInfo());
		config.valueProperty().addListener((observable, oldValue, value) -> {
			if (!filling) {
				showConfig(value);
			}
		});
		target.valueProperty().addListener((observable, oldValue, value) -> {
			if (!filling) {
				targetChanged(value);
			}
		});
		classes.textProperty().addListener((observable, oldValue, value) -> {
			targetScaling.setDisable(!classNames().isEmpty());
			onChange.run();
		});

		final var grid = new GridPane();
		grid.setHgap(8);
		grid.setVgap(6);
		grid.addRow(0, new Label("Repository"), new HBox(8, repository, loadInfo));
		HBox.setHgrow(repository, Priority.ALWAYS);
		grid.addRow(1, new Label("Config"), config);
		grid.addRow(2, new Label("Training split"), new HBox(8, split, new Label("rows"), samples, new Label(
				"shuffle seed"), seed));
		grid.addRow(3, new Label("Test split"), new HBox(8, testSplit, new Label("rows"), testSamples));
		grid.addRow(4, new Label("Input columns"), columns);
		grid.addRow(5, new Label("Target column"), target);
		grid.addRow(6, new Label("Classes"), classes);
		grid.addRow(7, new Label("Scaling"), new HBox(8, new Label("inputs"), inputScaling, new Label("target"),
				targetScaling));
		GridPane.setHgrow(columns, Priority.ALWAYS);
		GridPane.setHgrow(classes, Priority.ALWAYS);
		for (var node : grid.getChildren()) {
			if (node instanceof HBox box) {
				box.setAlignment(Pos.CENTER_LEFT);
			}
		}
		final var help = new Label("Rows are downloaded from the Hugging Face Parquet export into "
				+ DatasetCache.defaultDirectory() + " (at most " + DatasetCache.MAX_TABLE_ROWS + " rows per split) and"
				+ " shuffled with the seed. Without a test split, the first 'test rows' shuffled rows are held out."
				+ " Rows with a missing or non-numeric input are skipped. Scaling statistics come from the training"
				+ " rows; the app, ONNX and PMML exports all take raw column values.");
		help.setWrapText(true);
		help.setStyle("-fx-text-fill: #555;");
		getChildren().setAll(grid, info, help);
	}

	private static ComboBox<String> combo() {
		final var combo = new ComboBox<String>();
		combo.setEditable(true);
		combo.setPrefWidth(170);
		return combo;
	}

	private static TextField field(String text) {
		final var field = new TextField(text);
		field.setPrefColumnCount(6);
		return field;
	}

	void load(HuggingFaceTableDataset table) {
		filling = true;
		try {
			configs = List.of();
			statistics = Map.of();
			repository.setText(table.repository());
			config.getItems().setAll(table.config());
			config.setValue(table.config());
			split.getItems().clear();
			testSplit.getItems().clear();
			split.setValue(table.split());
			testSplit.setValue(table.testSplit());
			samples.setText(Integer.toString(table.samples()));
			testSamples.setText(Integer.toString(table.testSamples()));
			seed.setText(Long.toString(table.seed()));
			setColumns(table.inputColumns(), table.inputColumns());
			target.getItems().setAll(table.targetColumn());
			target.setValue(table.targetColumn());
			classes.setText(String.join(", ", table.classes()));
			inputScaling.setValue(table.inputScaling());
			targetScaling.setValue(table.targetScaling());
			info.setText("");
		} finally {
			filling = false;
		}
		onChange.run();
	}

	private void setColumns(List<String> names, List<String> selected) {
		columns.getChildren().clear();
		for (String name : names) {
			final var box = new CheckBox(name);
			box.setSelected(selected.contains(name));
			box.selectedProperty().addListener((observable, oldValue, value) -> onChange.run());
			columns.getChildren().add(box);
		}
	}

	List<String> inputColumns() {
		final String chosenTarget = text(target);
		return columns.getChildren().stream().map(CheckBox.class::cast).filter(CheckBox::isSelected)
				.map(CheckBox::getText).filter(name -> !name.equals(chosenTarget)).toList();
	}

	List<String> classNames() {
		return Arrays.stream(classes.getText().split("[,\\n]")).map(String::trim).filter(name -> !name.isEmpty())
				.toList();
	}

	int outputCount() {
		final int count = classNames().size();
		return count == 0 ? 1 : count;
	}

	int inputCount() {
		return inputColumns().size();
	}

	private static String text(ComboBox<String> combo) {
		final String typed = combo.getEditor().getText();
		final String text = typed == null || typed.isBlank() ? combo.getValue() : typed;
		return text == null ? "" : text.trim();
	}

	HuggingFaceTableDataset build(List<String> problems) {
		final String repo = repository.getText().trim();
		if (repo.isEmpty()) {
			problems.add("Enter a Hugging Face repository such as scikit-learn/iris");
			return null;
		}
		final String test = text(testSplit);
		long shuffle = 0;
		try {
			shuffle = Long.parseLong(seed.getText().trim());
		} catch (NumberFormatException exception) {
			problems.add("shuffle seed must be a whole number");
		}
		return new HuggingFaceTableDataset(repo, text(config), text(split), inputColumns(), text(target), classNames(),
				integer(samples, "training rows", problems), test.isEmpty() ? null : test,
				integer(testSamples, "test rows", problems), shuffle, inputScaling.getValue(), targetScaling.getValue());
	}

	private static int integer(TextField field, String label, List<String> problems) {
		try {
			return Integer.parseInt(field.getText().trim());
		} catch (NumberFormatException exception) {
			problems.add(label + " must be a whole number");
			return 0;
		}
	}

	// ---------------------------------------------------------------- Load info

	private record Loaded(List<HuggingFaceClient.ConfigInfo> configs,
			Map<String, HuggingFaceClient.ColumnStatistics> statistics, String config, String split) {
	}

	private void loadInfo() {
		final String repo = repository.getText().trim();
		if (!HuggingFaceClient.REPOSITORY.matcher(repo).matches()) {
			info.setText("Enter a repository such as scikit-learn/iris first.");
			return;
		}
		loadInfo.setDisable(true);
		info.setText("Asking Hugging Face about " + repo + "…");
		final var client = HuggingFaceClient.fromEnvironment();
		final String wantedConfig = text(config);
		final var task = new Task<Loaded>() {
			@Override
			protected Loaded call() throws Exception {
				final var found = client.info(repo);
				final var chosen = found.stream().filter(c -> c.config().equals(wantedConfig)).findFirst()
						.orElse(found.get(0));
				final String train = trainSplit(chosen);
				return new Loaded(found, statistics(client, repo, chosen.config(), train), chosen.config(), train);
			}
		};
		task.setOnSucceeded(event -> {
			loadInfo.setDisable(false);
			final var loaded = task.getValue();
			configs = loaded.configs();
			statistics = loaded.statistics();
			filling = true;
			try {
				config.getItems().setAll(configs.stream().map(HuggingFaceClient.ConfigInfo::config).toList());
				config.setValue(loaded.config());
			} finally {
				filling = false;
			}
			fill(configInfo(loaded.config()).orElseThrow(), loaded.split());
		});
		task.setOnFailed(event -> {
			loadInfo.setDisable(false);
			info.setText("Could not load dataset info: " + task.getException().getMessage());
		});
		Thread.ofVirtual().start(task);
	}

	private static Map<String, HuggingFaceClient.ColumnStatistics> statistics(HuggingFaceClient client, String repo,
			String config, String split) {
		final var map = new LinkedHashMap<String, HuggingFaceClient.ColumnStatistics>();
		try {
			client.statistics(repo, config, split).forEach(column -> map.put(column.name(), column));
		} catch (Exception exception) {
			// Statistics are optional (not computed for every dataset); class names can be typed in.
		}
		return map;
	}

	private static String trainSplit(HuggingFaceClient.ConfigInfo config) {
		final var splits = config.splits().keySet();
		return splits.contains("train") ? "train" : splits.stream().findFirst().orElse("train");
	}

	private Optional<HuggingFaceClient.ConfigInfo> configInfo(String name) {
		return configs.stream().filter(c -> c.config().equals(name)).findFirst();
	}

	/** Another config was picked: its columns and splits (statistics are only known for the loaded one). */
	private void showConfig(String name) {
		configInfo(name).ifPresent(found -> {
			statistics = Map.of();
			fill(found, trainSplit(found));
		});
	}

	private void fill(HuggingFaceClient.ConfigInfo found, String train) {
		filling = true;
		try {
			final var splits = found.splits().keySet();
			split.getItems().setAll(splits);
			testSplit.getItems().setAll(splits);
			split.setValue(train);
			final String test = splits.contains("test") ? "test" : splits.contains("validation") ? "validation" : null;
			testSplit.setValue(test);
			final long rows = found.splits().getOrDefault(train, -1L);
			if (test == null && rows > 0) {
				final long held = Math.max(1, Math.min(rows / 5, 1000));
				testSamples.setText(Long.toString(held));
				samples.setText(Long.toString(Math.min(rows - held, 2000)));
			} else if (rows > 0) {
				samples.setText(Long.toString(Math.min(rows, 2000)));
				testSamples.setText(Long.toString(Math.min(found.splits().getOrDefault(test, 500L), 500)));
			}
			final var names = found.features().stream().map(HuggingFaceClient.Feature::name).toList();
			final String chosenTarget = defaultTarget(found);
			target.getItems().setAll(names);
			target.setValue(chosenTarget);
			setColumns(names, names.stream().filter(name -> !name.equals(chosenTarget) && numeric(found, name)
					&& !ID_LIKE.matcher(name).matches()).toList());
		} finally {
			filling = false;
		}
		targetChanged(target.getValue());
		final var sizes = new StringBuilder();
		found.splits().forEach((name, count) -> sizes.append(sizes.isEmpty() ? "" : ", ").append(name)
				.append(count >= 0 ? " " + count : ""));
		final var skipped = found.features().stream().map(HuggingFaceClient.Feature::name)
				.filter(name -> ID_LIKE.matcher(name).matches()).toList();
		info.setText(found.config() + ": " + sizes + " rows. Numeric columns are selected as inputs"
				+ (skipped.isEmpty() ? "" : " (not " + String.join(", ", skipped) + ", which looks like a row id)")
				+ (statistics.isEmpty() ? "; no column statistics available" : "") + ".");
		onChange.run();
	}

	private String defaultTarget(HuggingFaceClient.ConfigInfo found) {
		final var features = found.features();
		return found.firstClassLabel().map(HuggingFaceClient.Feature::name)
				.or(() -> features.stream().map(HuggingFaceClient.Feature::name)
						.filter(name -> TARGET_LIKE.matcher(name).matches()).findFirst())
				.orElse(features.isEmpty() ? "" : features.get(features.size() - 1).name());
	}

	private boolean numeric(HuggingFaceClient.ConfigInfo found, String name) {
		final var stats = statistics.get(name);
		if (stats != null) {
			return stats.isNumeric() || "bool".equals(stats.type());
		}
		return found.feature(name).map(feature -> "Value".equals(feature.type()) && feature.dtype() != null
				&& feature.dtype().matches("u?int\\d*|float\\d*|double|bool")).orElse(false);
	}

	/** Fills the classes of a categorical target (ClassLabel names or the values counted by the statistics). */
	private void targetChanged(String name) {
		if (name == null) {
			return;
		}
		final var found = configInfo(text(config));
		final var feature = found.flatMap(c -> c.feature(name));
		final var stats = statistics.get(name);
		if (feature.isPresent() && feature.get().isClassLabel()) {
			classes.setText(String.join(", ", feature.get().classNames()));
		} else if (stats != null && stats.isCategorical()) {
			classes.setText(String.join(", ", stats.frequencies().keySet()));
		} else if (feature.isPresent() && "string".equals(feature.get().dtype())) {
			info.setText("'" + name + "' holds text but no statistics were available: type its class names.");
			classes.clear();
		} else if (found.isPresent()) {
			classes.clear();
		}
		for (var node : columns.getChildren()) {
			final var box = (CheckBox) node;
			box.setDisable(box.getText().equals(name));
			if (box.getText().equals(name)) {
				box.setSelected(false);
			}
		}
		onChange.run();
	}
}
