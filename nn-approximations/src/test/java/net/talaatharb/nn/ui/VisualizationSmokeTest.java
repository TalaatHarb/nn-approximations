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
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.config.NetworkSpec;
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
			await(stage, () -> hasLabel(stage, "Run logged") && mlflow.runs().size() == 3
					&& mlflow.runs().stream().noneMatch(run -> run.status.equals("RUNNING")));
			final var runs = mlflow.runs();
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
			return null;
		});
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
