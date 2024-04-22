package net.talaatharb.nn.structures;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

@Getter
@ToString
@RequiredArgsConstructor
public class TrainingData {

	private final float[][] input;
	private final float[][] output;

}
