package net.talaatharb.nn.ui;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;

import javafx.application.Platform;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.stage.Stage;
import net.talaatharb.nn.config.ActivationSpec;
import net.talaatharb.nn.config.ActivationType;
import net.talaatharb.nn.config.ExampleCatalog;
import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.config.NetworkSpec;
import net.talaatharb.nn.data.DatasetCache;
import net.talaatharb.nn.data.FakeHuggingFaceServer;
import net.talaatharb.nn.tracking.FakeMlflowServer;
import net.talaatharb.nn.tracking.MlflowSettings;

/** Opt-in because a desktop display is required. */
@EnabledIfSystemProperty(named = "javafx.smoke", matches = "true")
class VisualizationSmokeTest {

	@TempDir
	static Path models;

	private static FakeMlflowServer mlflow;

	@BeforeAll
	static void startToolkit() throws Exception {
		System.setProperty(ExampleCatalog.MODELS_DIR_PROPERTY, models.toString());
		mlflow = new FakeMlflowServer();
		System.setProperty("nn.mlflow.settings", models.resolve("settings/mlflow.json").toString());
		new MlflowSettings(true, mlflow.uri().toString(), "smoke").save();
		final var ready = new java.util.concurrent.CountDownLatch(1);
		Platform.startup(ready::countDown);
		assertTrue(ready.await(15, TimeUnit.SECONDS));
		Platform.setImplicitExit(false);
	}

	@AfterAll
	static void stopToolkit() {
		System.clearProperty(ExampleCatalog.MODELS_DIR_PROPERTY);
		System.clearProperty("nn.mlflow.settings");
		mlflow.close();
		Platform.exit();
	}

	@Test
	void canStepStopInferSwitchExamplesRandomizeAndTrainToCompletion() throws Exception {
		final var application = new NetworkVisualizationApplication();
		final var stage = fx(() -> {
			final var window = new Stage();
			application.start(window);
			return window;
		});
		try {
			fx(() -> {
				stage.getScene().getRoot().lookupAll(".slider").stream()
						.filter(Slider.class::isInstance).map(Slider.class::cast)
						.filter(slider -> slider.getMax() == 200).forEach(slider -> slider.setValue(0));
				button(stage, "Single update").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "update 1,"));
			assertTrue(fx(() -> button(stage, "Train").isDisabled()));
			assertTrue(fx(() -> button(stage, "Randomize").isDisabled()));
			Thread.sleep(150);
			assertFalse(fx(() -> hasLabel(stage, "update 2,")));
			fx(() -> {
				button(stage, "Single update").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "update 2,"));
			fx(() -> {
				button(stage, "Stop").fire();
				return null;
			});
			await(stage, () -> !button(stage, "Train").isDisabled());
			fx(() -> {
				button(stage, "Infer").fire();
				assertTrue(hasLabel(stage, "Input [0.0, 0.0]"));
				button(stage, "Train").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "Target reached:"));

			fx(() -> {
				final var seed = stage.getScene().getRoot().lookupAll(".text-field").stream()
						.filter(TextField.class::isInstance).map(TextField.class::cast)
						.filter(field -> "random".equals(field.getPromptText())).findFirst().orElseThrow();
				seed.setText("123");
				button(stage, "Randomize").fire();
				final var first = application.currentWeights();
				assertTrue(hasLabel(stage, "seed 123"));
				button(stage, "Randomize").fire();
				assertArrayEquals(first, application.currentWeights());
				seed.setText("");
				button(stage, "Randomize").fire();
				assertFalse(java.util.Arrays.deepEquals(first, application.currentWeights()));
				return null;
			});

			fx(() -> {
				final ComboBox<ExampleCatalog.Entry> selector = modelSelector(stage);
				selector.setValue(selector.getItems().stream().filter(entry -> entry.id().equals("square"))
						.findFirst().orElseThrow());
				assertTrue(hasLabel(stage, "Input [-1.0]"));
				button(stage, "Train").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "Target reached:"));
			assertTrue(fx(() -> !button(stage, "Infer").isDisabled()));
			await(stage, () -> hasLabel(stage, "Run logged") && smokeRuns().size() == 3
					&& smokeRuns().stream().noneMatch(run -> run.status.equals("RUNNING")));
			final var runs = smokeRuns();
			assertEquals(List.of("KILLED", "FINISHED", "FINISHED"), runs.stream().map(run -> run.status).toList());
			assertEquals(List.of("xor", "xor", "square"), runs.stream().map(run -> run.params.get("model.id")).toList());
			final var xorRun = runs.get(1);
			assertEquals("1.0", xorRun.metrics("accuracy").get(0).path("value").asText());
			assertFalse(xorRun.metrics("loss").isEmpty());
			assertTrue(xorRun.params.get("weights.origin").startsWith("continued training from"));
			assertTrue(runs.get(2).params.get("weights.origin").contains("seed"));
			assertTrue(mlflow.artifacts.keySet().stream()
					.anyMatch(path -> path.endsWith(runs.get(2).id + "/artifacts/model/model.onnx")));

			final Path exported = models.resolve("export-check.out");
			final Path onnx = models.resolve("export-check.onnx");
			final Path pmml = models.resolve("export-check.pmml");
			fx(() -> {
				application.exportTo(exported, NetworkVisualizationApplication.ExportFormat.WITH_WEIGHTS);
				application.exportTo(onnx, NetworkVisualizationApplication.ExportFormat.ONNX);
				application.exportTo(pmml, NetworkVisualizationApplication.ExportFormat.PMML);
				return null;
			});
			final var reloaded = ModelJson.read(Files.readString(exported));
			assertNotNull(reloaded.weights());
			assertEquals("y = x^2", reloaded.name());
			assertTrue(Files.size(onnx) > 100);
			assertTrue(Files.readString(pmml).contains("<NeuralNetwork"));

			final String shownJson = fx(() -> {
				button(stage, "View JSON").fire();
				final var viewer = javafx.stage.Window.getWindows().stream().filter(JsonViewer.class::isInstance)
						.map(JsonViewer.class::cast).findFirst().orElseThrow();
				final var area = (JsonCodeArea) viewer.getScene().getRoot().lookup(".json-area");
				final String text = area.getText();
				final boolean highlighted = area.getStyleSpans(0, area.getLength()).getSpanCount() > 10;
				viewer.close();
				return highlighted ? text : "not highlighted";
			});
			assertEquals(ModelJson.write(reloaded), shownJson);
		} finally {
			fx(() -> {
				application.stop();
				stage.close();
				return null;
			});
		}
	}

	@Test
	void imageClassificationLoadsFromHuggingFaceInfersAndTrains() throws Exception {
		final Path cache = models.resolve("hf-cache");
		System.setProperty(DatasetCache.DIRECTORY_PROPERTY, cache.toString());
		final var hub = new FakeHuggingFaceServer();
		hub.install();
		final var application = new NetworkVisualizationApplication();
		Stage stage = null;
		try {
			new ExampleCatalog(models).save("shapes-smoke", ModelJson.read(FakeHuggingFaceServer.SHAPES_JSON));
			stage = fx(() -> {
				final var window = new Stage();
				application.start(window);
				return window;
			});
			final var window = stage;
			fx(() -> {
				final ComboBox<ExampleCatalog.Entry> selector = modelSelector(window);
				selector.setValue(selector.getItems().stream().filter(entry -> entry.id().equals("shapes-smoke"))
						.findFirst().orElseThrow());
				return null;
			});
			await(window, () -> hasLabel(window, "60 training + 30 test images from test/shapes"));
			fx(() -> {
				final var view = (ImageClassificationView) window.getScene().getRoot().lookupAll("*").stream()
						.filter(ImageClassificationView.class::isInstance).findFirst().orElseThrow();
				assertTrue(view.isVisible());
				assertTrue(view.visibleThumbnails() > 0);
				assertTrue(hasLabel(window, "horizontal"));
				final var input = window.getScene().getRoot().lookupAll(".text-field").stream()
						.filter(TextField.class::isInstance).map(TextField.class::cast)
						.filter(field -> "test:0".equals(field.getText())).findFirst().orElseThrow();
				input.setText("test:1");
				button(window, "Infer").fire();
				assertTrue(hasLabel(window, "test:1 (label vertical)"));
				button(window, "Train").fire();
				return null;
			});
			await(window, () -> hasLabel(window, "Epoch limit reached:") || hasLabel(window, "Target reached:"));
			await(window, () -> mlflow.runs().stream().anyMatch(run -> "shapes-smoke".equals(run.params.get("model.id"))
					&& run.status.equals("FINISHED")));
			final var run = mlflow.runs().stream().filter(item -> "shapes-smoke".equals(item.params.get("model.id")))
					.findFirst().orElseThrow();
			assertEquals("huggingFace", run.params.get("dataset.type"));
			assertTrue(Double.parseDouble(run.metrics("test_accuracy").get(0).path("value").asText()) >= 0.6,
					"the strokes are easy to tell apart");
			assertTrue(hub.requests.stream().anyMatch(path -> path.contains("/files/test/")));
		} finally {
			final var window = stage;
			fx(() -> {
				application.stop();
				if (window != null) {
					window.close();
				}
				return null;
			});
			hub.close();
			FakeHuggingFaceServer.uninstall();
			System.clearProperty(DatasetCache.DIRECTORY_PROPERTY);
		}
	}

	private static final String FLOWERS_JSON = """
			{
			  "name" : "Flowers",
			  "task" : "classification",
			  "network" : { "inputs" : 4, "layers" : [ { "neurons" : 6, "activation" : "tanh" },
			    { "neurons" : 3, "activation" : "sigmoid" } ] },
			  "initialization" : { "scheme" : "xavierUniform", "seed" : 3 },
			  "training" : { "learningRate" : 0.2, "targetLoss" : 0.0001, "maxEpochs" : 60, "shuffleSeed" : 1 },
			  "dataset" : {
			    "type" : "huggingFaceTable",
			    "repository" : "test/flowers",
			    "inputColumns" : [ "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" ],
			    "targetColumn" : "Species",
			    "classes" : [ "setosa", "versicolor", "virginica" ],
			    "samples" : 100,
			    "testSamples" : 15,
			    "seed" : 5
			  }
			}
			""";

	@Test
	void tableDatasetLoadsFromHuggingFaceInfersTrainsAndEdits() throws Exception {
		final Path cache = models.resolve("hf-table-cache");
		System.setProperty(DatasetCache.DIRECTORY_PROPERTY, cache.toString());
		final var hub = new FakeHuggingFaceServer();
		hub.install();
		final var application = new NetworkVisualizationApplication();
		Stage stage = null;
		try {
			new ExampleCatalog(models).save("flowers-smoke", ModelJson.read(FLOWERS_JSON));
			stage = fx(() -> {
				final var window = new Stage();
				application.start(window);
				return window;
			});
			final var window = stage;
			fx(() -> {
				final ComboBox<ExampleCatalog.Entry> selector = modelSelector(window);
				selector.setValue(selector.getItems().stream().filter(entry -> entry.id().equals("flowers-smoke"))
						.findFirst().orElseThrow());
				return null;
			});
			await(window, () -> hasLabel(window, "test rows from test/flowers (scaled)"));
			fx(() -> {
				final var input = window.getScene().getRoot().lookupAll(".text-field").stream()
						.filter(TextField.class::isInstance).map(TextField.class::cast)
						.filter(field -> field.getText().split(",").length == 4).findFirst().orElseThrow();
				final var results = window.getScene().getRoot().lookupAll(".text-area").stream()
						.filter(node -> node.getStyle().contains("monospace")).findFirst().orElseThrow();
				javafx.scene.Parent parent = results.getParent();
				while (!(parent instanceof javafx.scene.control.ScrollPane)) {
					parent = parent.getParent();
				}
				final var scroll = (javafx.scene.control.ScrollPane) parent;
				window.setHeight(500);
				window.getScene().getRoot().applyCss();
				window.getScene().getRoot().layout();
				assertTrue(scroll.getContent().getBoundsInLocal().getHeight() > scroll.getViewportBounds().getHeight(),
						"a short window scrolls the charts panel");
				assertTrue(results.getBoundsInParent().getHeight() >= 160, "the results table keeps its height");
				scroll.setVvalue(1);
				window.setHeight(900);
				input.setText("5.1, 3.5, 1.4, 0.2");
				button(window, "Infer").fire();
				assertTrue(hasLabel(window, "Input [5.1, 3.5, 1.4, 0.2] ->"));
				input.setText("test:2");
				button(window, "Infer").fire();
				assertTrue(hasLabel(window, "test:2 ["));
				assertTrue(hasLabel(window, "(target "));
				window.getScene().getRoot().lookupAll(".slider").stream()
						.filter(javafx.scene.control.Slider.class::isInstance)
						.map(javafx.scene.control.Slider.class::cast).forEach(slider -> slider.setValue(0));
				button(window, "Train").fire();
				return null;
			});
			await(window, () -> hasLabel(window, "Epoch limit reached:") || hasLabel(window, "Target reached:"));
			await(window, () -> mlflow.runs().stream().anyMatch(run -> "flowers-smoke".equals(run.params.get("model.id"))
					&& run.status.equals("FINISHED")));
			final var run = mlflow.runs().stream().filter(item -> "flowers-smoke".equals(item.params.get("model.id")))
					.findFirst().orElseThrow();
			assertEquals("huggingFaceTable", run.params.get("dataset.type"));
			assertEquals("Species", run.params.get("dataset.target"));
			assertEquals("held out from train", run.params.get("dataset.test_split"));
			assertTrue(Double.parseDouble(run.metrics("test_accuracy").get(0).path("value").asText()) >= 0.6,
					"species separate well on petal sizes");
			assertTrue(run.metrics("accuracy").size() > 0);

			fx(() -> {
				final var editor = new ModelEditorPane();
				final var flowers = ModelJson.read(FLOWERS_JSON);
				editor.load(flowers);
				assertEquals(ModelJson.write(flowers), ModelJson.write(editor.build()));
				final var houses = ExampleCatalog.builtIn("california_housing");
				editor.load(houses);
				assertEquals(ModelJson.write(houses), ModelJson.write(editor.build()));
				return null;
			});
			final var form = fx(() -> {
				final var editor = new ModelEditorPane();
				editor.load(ModelJson.read(FLOWERS_JSON));
				final var table = editor.tableForm();
				table.repository.setText(FakeHuggingFaceServer.HOUSES);
				table.loadInfo.fire();
				return new Object[] { editor, table };
			});
			final var editor = (ModelEditorPane) form[0];
			final var table = (HuggingFaceTableForm) form[1];
			awaitFx(() -> !table.loadInfo.isDisabled() && !table.info.getText().startsWith("Asking"));
			fx(() -> {
				assertTrue(table.classNames().isEmpty(), table.info.getText());
				assertEquals("test", table.testSplit.getValue());
				final var built = editor.build();
				assertEquals(ModelDefinition.Task.REGRESSION, built.task());
				assertEquals(1, built.network().layers().get(built.network().layers().size() - 1).neurons());
				final var dataset = (net.talaatharb.nn.config.HuggingFaceTableDataset) built.dataset();
				assertEquals(List.of("rooms", "age"), dataset.inputColumns());
				assertEquals("price", dataset.targetColumn());
				table.repository.setText(FakeHuggingFaceServer.FLOWERS);
				table.loadInfo.fire();
				return null;
			});
			awaitFx(() -> !table.loadInfo.isDisabled() && table.info.getText().startsWith("default"));
			fx(() -> {
				assertEquals(FakeHuggingFaceServer.SPECIES, table.classNames(), table.info.getText());
				final var built = editor.build();
				assertEquals(ModelDefinition.Task.CLASSIFICATION, built.task());
				final var dataset = (net.talaatharb.nn.config.HuggingFaceTableDataset) built.dataset();
				assertEquals(FakeHuggingFaceServer.FLOWER_FEATURES, dataset.inputColumns(), "Id is left out");
				assertEquals(3, built.network().layers().get(built.network().layers().size() - 1).neurons());
				return null;
			});
		} finally {
			final var window = stage;
			fx(() -> {
				application.stop();
				if (window != null) {
					window.close();
				}
				return null;
			});
			hub.close();
			FakeHuggingFaceServer.uninstall();
			System.clearProperty(DatasetCache.DIRECTORY_PROPERTY);
		}
	}

	private static void awaitFx(BooleanSupplier condition) throws Exception {
		final long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
		while (!fx(condition::getAsBoolean)) {
			if (System.nanoTime() > deadline) {
				fail("Timed out waiting for the editor");
			}
			Thread.sleep(50);
		}
	}

	@Test
	void editorAddsLayersAndRejectsInvalidInput() throws Exception {
		fx(() -> {
			final var editor = new ModelEditorPane();
			final var xor = ExampleCatalog.builtIn("xor");
			editor.load(xor.withWeights(xor.createNetwork(1).copyWeights()));
			assertEquals(xor.network(), editor.build().network());
			assertNotNull(editor.build().weights(), "Unchanged topology keeps weights");
			editor.addHiddenLayer(new NetworkSpec.LayerSpec(3, ActivationSpec.of(ActivationType.RELU)));
			final var built = editor.build();
			assertEquals(3, built.network().layers().size());
			assertEquals("relu", built.network().layers().get(1).activation().type());
			assertNull(built.weights(), "Changed topology drops stored weights");

			final var sine = ExampleCatalog.builtIn("sine");
			editor.load(sine);
			assertEquals(ModelJson.write(sine), ModelJson.write(editor.build()));

			final var mnist = ExampleCatalog.builtIn("mnist");
			editor.load(mnist);
			assertEquals(ModelJson.write(mnist), ModelJson.write(editor.build()));
			editor.load(xor);
			assertEquals(ModelDefinition.Task.CLASSIFICATION, editor.build().task());
			return null;
		});
	}

	private static List<FakeMlflowServer.Run> smokeRuns() {
		return mlflow.runs().stream().filter(run -> !"shapes-smoke".equals(run.params.get("model.id"))
				&& !"flowers-smoke".equals(run.params.get("model.id"))).toList();
	}

	@SuppressWarnings("unchecked")
	private static ComboBox<ExampleCatalog.Entry> modelSelector(Stage stage) {
		return stage.getScene().getRoot().lookupAll(".combo-box").stream().filter(ComboBox.class::isInstance)
				.map(ComboBox.class::cast)
				.filter(box -> !box.getItems().isEmpty() && box.getItems().get(0) instanceof ExampleCatalog.Entry)
				.findFirst().orElseThrow();
	}

	private static Button button(Stage stage, String text) {
		return stage.getScene().getRoot().lookupAll(".button").stream()
				.filter(Button.class::isInstance).map(Button.class::cast)
				.filter(button -> button.getText().equals(text)).findFirst().orElseThrow();
	}

	private static boolean hasLabel(Stage stage, String text) {
		return stage.getScene().getRoot().lookupAll(".label").stream()
				.filter(Label.class::isInstance).map(Label.class::cast)
				.anyMatch(label -> label.getText() != null && label.getText().contains(text));
	}

	private static void await(Stage stage, BooleanSupplier condition) throws Exception {
		long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
		while (!fx(condition::getAsBoolean)) {
			if (System.nanoTime() > deadline) {
				fail("Timed out waiting for the desktop state: " + fx(() ->
						stage.getScene().getRoot().lookupAll(".label").stream()
								.filter(Label.class::isInstance).map(Label.class::cast)
								.map(Label::getText).toList()));
			}
			Thread.sleep(30);
		}
	}

	private static <T> T fx(Callable<T> action) throws Exception {
		final var task = new FutureTask<>(action);
		Platform.runLater(task);
		return task.get(15, TimeUnit.SECONDS);
	}
}
