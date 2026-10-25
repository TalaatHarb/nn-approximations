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

Select XOR, AND, OR, NAND, NOR, a linear function, square, or sine. Mathematical examples use 41 samples on `[-1, 1]`; logic examples use the complete binary truth table.

- **Train** continues from the current weights; **Reset** restores reproducible seed-42 weights.
- **Pause / Resume**, **Single update**, and **Stop** control training. Single update also starts a paused training session. The delay slider sets milliseconds per update; use zero to converge quickly.
- Inspect neuron activations, pre-activation sums and biases, signed/weighted connections, and before/after-update weight changes via hover. The checkbox switches between states of the last update.
- The loss chart uses dataset MSE evaluated with each completed epoch's final weights. The second chart and table compare predictions with targets; logic outputs are classified at `0.5`.
- When training stops, enter comma-separated inputs and click **Infer**. Use the layer slider or **Next layer** to reveal hidden activations progressively. This is a reveal of one completed inference, not additional training.

Training runs on a background thread. Inference and reset are disabled until that worker exits. The UI consumes detached snapshots at up to 30 frames/second, coalescing intermediate frames instead of queuing every update; listeners still receive every SGD update. Chart history is bounded to 600 rendered epoch points. At high speed, some epochs will not appear on the chart.

## Observation API

The numerical core has no JavaFX dependency. Existing `apply`, layer getters and the original `train` overload remain available.

```java
var example = ExampleProblem.XOR;
var network = example.newNetwork();
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
new StochasticGradientDescent(example.learningRate()).train(
    network, example.data(), example.threshold(), 5000,
    new MeanSquareError(), listener, () -> false);
```

Classes above live under `net.talaatharb.nn.examples`, `.algorithms`, and `.metrics`. `beforeUpdate` can gate execution for interactive stepping. Callbacks run synchronously on the training thread; callback exceptions propagate. Cancellation (the supplier or thread interruption) throws `CancellationException` and retains completed updates. Do not interrogate or mutate a live network from another thread; pass snapshots instead.

`InspectableNetwork` and `NetworkSnapshot` in `.visualization` form the extension boundary. The graph format does not require forward-only edges, so a future RBM adapter can expose visible/hidden units and bidirectional connections without rewriting the renderer. RBM sampling, reconstruction phases and contrastive-divergence training are **not implemented**; SGD remains specifically feed-forward.

## Build and checks

The Maven setup targets Java 21, explicitly configures Lombok annotation processing (including newer JDK support), and uses the real desktop launcher rather than the template's nonexistent main class. Unused MapStruct/Mockito dependencies and the overwriting fat-JAR assembly were removed. JaCoCo reports are generated in `target/site/jacoco`; the previous template's blanket per-package coverage gate and unused mutation/static-analysis configuration are not part of the default build.

Tests cover finite-difference backpropagation across multiple hidden layers and outputs, immutable snapshots, cancellation, validation, truth tables and convergence/interpolation of every viewer example. An optional desktop smoke test exercises stepping, stopping, inference, example switching and convergence:

```powershell
mvn "-Dtest=VisualizationSmokeTest" "-Djavafx.smoke=true" test
```

The dependency audit is opt-in because it requires external vulnerability feeds:

```powershell
mvn -Psecurity-audit verify
```

If Maven reports an unreachable corporate mirror, correct your Maven `settings.xml` or use `mvn -s <settings-file> ...` with settings appropriate to your network; the project itself uses Maven Central.
