# nn-approximations
Building a project for building neural network approximations for mathematical functions and visualizing the process of training and inference

## Run the desktop application

Install JDK 21 or newer and Maven 3.9+, then run from the repository root:

```powershell
Set-Location nn-approximations
mvn clean verify
mvn javafx:run
```

JavaFX dependencies and platform-native libraries are resolved by Maven; no separate JavaFX SDK is needed. A graphical desktop is required to run the viewer, but not for the normal test suite. The packaged JAR is not a standalone distribution: launch with `javafx:run`, which supplies the JavaFX runtime.

## Models and examples

Every example is a JSON **model definition**. Eight built-ins ship as resources (`src/main/resources/examples`: XOR, AND, OR, NAND, NOR, linear, square, sine). User models are stored as `<id>.json` in `~/.nn-approximations/models` (override with the `nn.models.dir` Java system property). A user file with a built-in id (e.g. `xor.json`) overrides that built-in; **Restore built-in** deletes the override. Invalid files are skipped and reported in the status line.

- **New / Edit / Duplicate** open the model editor. Editing a built-in saves a modified copy; New and Duplicate save under a fresh id derived from the name.
- **Import** copies a JSON file into the model directory after validation. **Export** writes the configuration, optionally with the network's current weights.
- The editor has three tabs:
  - **Model**: name, description, task (regression/classification), any number of hidden layers (neurons, activation and activation parameters, reorder/remove), output activation, weight initialization and training defaults. Input and output layer sizes follow the dataset.
  - **Data**: either a *function* (input variables with min/max/sample count, sampled as a grid, plus one expression per output, e.g. `sin(pi*x)`, `x1*x2`, `xor(a, b)`) or a *table* of rows such as `0, 1 -> 1`.
  - **JSON**: edit the raw definition and apply it back to the form.
  Stored weights are kept only when the topology is unchanged.

```json
{
  "modelType" : "feedForward",
  "formatVersion" : 1,
  "name" : "XOR",
  "task" : "classification",
  "network" : { "inputs" : 2, "layers" : [ { "neurons" : 4, "activation" : "tanh" }, { "neurons" : 1, "activation" : "sigmoid" } ] },
  "initialization" : { "scheme" : "uniform", "min" : -1.0, "max" : 1.0, "seed" : 42 },
  "training" : { "learningRate" : 0.3, "targetLoss" : 0.005, "maxEpochs" : 5000, "shuffleSeed" : 42 },
  "dataset" : { "type" : "table", "inputNames" : [ "a", "b" ], "outputNames" : [ "xor" ],
                "inputs" : [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ], "outputs" : [ [ 0 ], [ 1 ], [ 1 ], [ 0 ] ] }
}
```

- `network.layers` lists hidden layers followed by the output layer. Activations: `linear`, `sigmoid`, `tanh`, `relu`, `leakyRelu`, `elu`, and parameterized `{"type":"parametricRelu","alpha":0.1}`, `{"type":"scaledElu","alpha":1}`, `{"type":"piecewiseLinear","xMin":-1,"xMax":1}`, `{"type":"step","threshold":0}` (step has zero gradient, so it is not trainable).
- `initialization.scheme`: `uniform` (biases and weights in `[min, max)`), `xavierUniform` or `heUniform` (zero biases). Omit `seed` for a new random seed each reset; omit `training.shuffleSeed` for a different sample order on each run.
- A function dataset looks like `{"type":"function","inputs":[{"name":"x","min":-1,"max":1,"samples":41}],"outputs":["x^2"]}`. Expressions support `+ - * / % ^`, `pi`, `e`, common math functions (`sin`, `exp`, `log`, `sqrt`, `abs`, `min`, `max`, `pow`, `atan2`, ...) and logic helpers (`not`, `and`, `or`, `xor`, `nand`, `nor`, `step`) that treat values `>= 0.5` as true.
- `weights` is optional: `[layer][neuron][bias, w1..wn]`. When present, **Reset to model** restores it instead of initializing.
- Unknown properties and unsupported `modelType`s are rejected. `modelType` reserves room for other model kinds (such as RBMs) later.

## Viewer controls

- **Train** continues from the current weights; **Pause / Resume**, **Single update** and **Stop** control training. Single update also starts a paused session. The delay slider sets milliseconds per update; use zero to converge quickly. Rate, epochs and target MSE default to the model's training settings.
- **Weights** row: pick a scheme, range and seed, then **Randomize**. A blank seed draws a new one each time and the status line reports it so the result can be reproduced. **Reset to model** rebuilds from the model's stored weights or initialization. **Store weights in model** saves the current weights into the model file; **Clear stored weights** removes them.
- The network canvas scales to any layer size. Edge width and opacity are relative to the largest value shown. Blue is positive and red is negative; the node fill shows activation and the ring shows bias. Switch **Edges** to *Last update change* to see where SGD moved weights. Hover a neuron to highlight its connections and list z, bias, deltas and its strongest incoming/outgoing weights.
- Use the layer slider or **Next layer** to reveal hidden activations of one inference progressively. When idle, enter comma-separated inputs and click **Infer**.
- The loss chart shows dataset MSE per epoch, optionally on a log10 scale. Long runs are downsampled, so the full history stays visible.
- Single-input models plot target vs. prediction. Two-input models show a live heatmap of the selected output over the input plane with training points overlaid and, for classification, the 0.5 decision boundary. Other input counts plot by sample index. The table lists all inputs, targets and outputs; classification tasks also show the class at 0.5.

Training runs on a background thread. Model management, weight changes and inference are disabled until that worker exits. The UI consumes detached snapshots at up to 30 frames/second and coalesces intermediate frames; listeners still receive every SGD update. The heatmap is recomputed on the training thread at most every 120 ms.

## Observation API

The numerical core has no JavaFX dependency. Existing `apply`, layer getters and the original `train` overload remain available.

```java
var model = ExampleCatalog.builtIn("xor");          // or ModelJson.read(path).requireValid()
var network = model.createNetwork(42);              // stored weights, or initialization (42 if no seed)
var data = model.dataset().toTrainingData();
var snapshot = network.inspect(new float[] {0, 1});
// snapshot.layers(): input, hidden and output activations, z values and biases
// snapshot.connections(): topology and signed weights; all lists are immutable

var listener = new TrainingListener() {
    @Override
    public void onUpdate(TrainingUpdate update) {
        // epoch, step, sample index, targets, sample losses and before/after snapshots
    }

    @Override
    public void onEpoch(int epoch, float datasetLoss) {
        // Persist metrics or marshal them to a UI.
    }
};
var training = model.training();
new StochasticGradientDescent(training.learningRate(), training.shuffleRandom()).train(
    network, data, training.targetLoss(), training.maxEpochs(),
    new MeanSquareError(), listener, () -> false);
ModelJson.write(model.withWeights(network.copyWeights()), Path.of("xor-trained.json"));
```
Classes above live under `net.talaatharb.nn.config`, `.algorithms`, and `.metrics`. `beforeUpdate` can gate execution for interactive stepping. Callbacks run synchronously on the training thread; callback exceptions propagate. Cancellation (the supplier or thread interruption) throws `CancellationException` and retains completed updates. Do not interrogate or mutate a live network from another thread; pass snapshots instead.

`InspectableNetwork` and `NetworkSnapshot` in `.visualization` form the extension boundary. The graph format does not require forward-only edges, so a future RBM adapter can expose visible/hidden units and bidirectional connections without rewriting the renderer. RBM sampling, reconstruction phases and contrastive-divergence training are **not implemented**; SGD remains specifically feed-forward.

## Build and checks

The Maven setup targets Java 21, explicitly configures Lombok annotation processing (including newer JDK support), and uses the real desktop launcher rather than the template's nonexistent main class. Unused MapStruct/Mockito dependencies and the overwriting fat-JAR assembly were removed. JaCoCo reports are generated in `target/site/jacoco`; the previous template's blanket per-package coverage gate and unused mutation/static-analysis configuration are not part of the default build.

Tests cover finite-difference backpropagation across multiple hidden layers and outputs, immutable snapshots, cancellation, validation, truth tables, the expression language, JSON round trips and validation, initializer reproducibility, the model catalog, and convergence/interpolation of every built-in example. An optional desktop smoke test exercises stepping, stopping, inference, model switching, seeded randomization, export, the editor and convergence:

```powershell
mvn "-Dtest=VisualizationSmokeTest" "-Djavafx.smoke=true" test
```

The dependency audit is opt-in because it requires external vulnerability feeds:

```powershell
mvn -Psecurity-audit verify
```

If Maven reports an unreachable corporate mirror, correct your Maven `settings.xml` or use `mvn -s <settings-file> ...` with settings appropriate to your network; the project itself uses Maven Central.
