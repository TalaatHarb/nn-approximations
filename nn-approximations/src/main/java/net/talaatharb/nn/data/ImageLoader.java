package net.talaatharb.nn.data;

import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;

import javax.imageio.ImageIO;

/** Decodes an image file into normalized pixel inputs (0..1), resizing when needed. */
public final class ImageLoader {

	private ImageLoader() {
	}

	/**
	 * @return {@code width * height * channels} values, row-major, channels interleaved (RGB) per pixel
	 */
	public static float[] load(Path file, int width, int height, boolean grayscale) {
		final BufferedImage image;
		try {
			image = ImageIO.read(file.toFile());
		} catch (IOException exception) {
			throw new UncheckedIOException("Could not read image " + file + ": " + exception.getMessage(), exception);
		}
		if (image == null) {
			throw new IllegalArgumentException("Unsupported image format: " + file
					+ " (PNG, JPEG, GIF and BMP are supported)");
		}
		return pixels(image, width, height, grayscale);
	}

	public static float[] pixels(BufferedImage source, int width, int height, boolean grayscale) {
		BufferedImage image = source;
		if (source.getWidth() != width || source.getHeight() != height) {
			image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			final var graphics = image.createGraphics();
			graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			graphics.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
			graphics.drawImage(source, 0, 0, width, height, null);
			graphics.dispose();
		}
		final int channels = grayscale ? 1 : 3;
		final float[] values = new float[width * height * channels];
		final boolean gray = image.getType() == BufferedImage.TYPE_BYTE_GRAY;
		final var raster = image.getRaster();
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				final int index = (y * width + x) * channels;
				if (gray) {
					final float value = raster.getSample(x, y, 0) / 255f;
					for (int c = 0; c < channels; c++) {
						values[index + c] = value;
					}
					continue;
				}
				final int rgb = image.getRGB(x, y);
				final int r = rgb >> 16 & 0xFF;
				final int g = rgb >> 8 & 0xFF;
				final int b = rgb & 0xFF;
				if (grayscale) {
					values[index] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f;
				} else {
					values[index] = r / 255f;
					values[index + 1] = g / 255f;
					values[index + 2] = b / 255f;
				}
			}
		}
		return values;
	}
}
