package net.talaatharb.nn.algorithms;

/**
 * Callbacks execute synchronously on the training thread. Exceptions propagate to
 * the caller; consumers must marshal snapshots to their UI thread themselves.
 */
public interface TrainingListener {

	TrainingListener NONE = new TrainingListener() { };

	default void beforeUpdate(int epoch, long step, int sampleIndex) {
	}

	/**
	 * Whether update {@code step} should be observed: snapshots are captured before/after it and
	 * {@link #onUpdate} is called. Snapshots of large networks are costly, so a UI can sample.
	 */
	default boolean observeUpdate(long step) {
		return true;
	}

	default void onUpdate(TrainingUpdate update) {
	}

	/** Loss evaluated over the entire dataset using the epoch's final weights. */
	default void onEpoch(int epoch, float loss) {
	}
}
