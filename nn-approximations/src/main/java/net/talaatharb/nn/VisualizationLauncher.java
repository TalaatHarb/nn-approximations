package net.talaatharb.nn;

import javafx.application.Application;
import net.talaatharb.nn.ui.NetworkVisualizationApplication;

public final class VisualizationLauncher {

	private VisualizationLauncher() {
	}

	public static void main(String[] args) {
		Application.launch(NetworkVisualizationApplication.class, args);
	}
}
