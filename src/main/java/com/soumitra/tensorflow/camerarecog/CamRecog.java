package com.soumitra.tensorflow.camerarecog;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.UInt8;

@Component("camRecog")
public class CamRecog {

	private static final Logger logger = LoggerFactory.getLogger(CamRecog.class);

	public String execute(byte[] imageData, ClassLoader classLoader) throws Exception {
		if (imageData.length > 0 && classLoader == null) {
			logger.error(
					"This program uses a pre-trained inception model to identify JPEG images. Takes byte stream of images and compares with the model.");
			logger.error("TensorFlow version: {} ", TensorFlow.version() != null ? TensorFlow.version() : 0);
			System.exit(1);
		}
		List<String> labels = new ArrayList<>();
		try (InputStream modelAsStream = classLoader
				.getResourceAsStream("models/inception5h/tensorflow_inception_graph.pb");
				InputStream labelsAsStream = classLoader
						.getResourceAsStream("models/inception5h/imagenet_comp_graph_label_strings.txt");
				Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageData);
				Scanner scanner = new Scanner(labelsAsStream);) {
			byte[] graphDef = IOUtils.toByteArray(modelAsStream);
			while (scanner.hasNextLine()) {
				labels.add(scanner.nextLine());
			}
			float[] labelProbabilities = executeInceptionGraph(graphDef, image);
			int bestMatchCaseLabelIndex = maxIndex(labelProbabilities);
			String result = String.format("Probable match: %s (%.2f percentage)", labels.get(bestMatchCaseLabelIndex),
					labelProbabilities[bestMatchCaseLabelIndex] * 100f);
			logger.info(result);
			return result;
		} catch (Exception e) {
			logger.error(e.getMessage(), e);
			throw new Exception(e.getMessage(), e);
		}

	}

	private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
		try (Graph g = new Graph()) {
			GraphBuilder b = new GraphBuilder(g);
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to
			// float using (value - Mean)/Scale.
			final int H = 224;
			final int W = 224;
			final float mean = 117f;
			final float scale = 1f;

			// Since the graph is being constructed once per execution here, we can use a
			// constant for the
			// input image. If the graph were to be re-used for multiple input images, a
			// placeholder would
			// have been more appropriate.
			final Output<String> input = b.constant("input", imageBytes);
			final Output<Float> output = b
					.div(b.sub(
							b.resizeBilinear(b.expandDims(b.cast(b.decodeJpeg(input, 3), Float.class),
									b.constant("make_batch", 0)), b.constant("size", new int[] { H, W })),
							b.constant("mean", mean)), b.constant("scale", scale));
			try (Session s = new Session(g)) {
				return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
			}
		}
	}

	private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
		try (Graph g = new Graph()) {
			g.importGraphDef(graphDef);
			try (Session s = new Session(g);
					Tensor<Float> result = s.runner().feed("input", image).fetch("output").run().get(0)
							.expect(Float.class)) {
				final long[] rshape = result.shape();
				if (result.numDimensions() != 2 || rshape[0] != 1) {
					logger.error(
							"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape {} ",
							Arrays.toString(rshape));
					throw new RuntimeException(String.format(
							"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
							Arrays.toString(rshape)));
				}
				int nlabels = (int) rshape[1];
				return result.copyTo(new float[1][nlabels])[0];
			}
		}
	}

	private static int maxIndex(float[] probabilities) {
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i) {
			if (probabilities[i] > probabilities[best]) {
				best = i;
			}
		}
		return best;
	}

	// In the fullness of time, equivalents of the methods of this class should be
	// auto-generated from
	// the OpDefs linked into libtensorflow_jni.so. That would match what is done in
	// other languages
	// like Python, C++ and Go.
	static class GraphBuilder {

		private Graph g;

		GraphBuilder(Graph g) {
			this.g = g;
		}

		Output<Float> div(Output<Float> x, Output<Float> y) {
			return binaryOp("Div", x, y);
		}

		<T> Output<T> sub(Output<T> x, Output<T> y) {
			return binaryOp("Sub", x, y);
		}

		<T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
			return binaryOp3("ResizeBilinear", images, size);
		}

		<T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
			return binaryOp3("ExpandDims", input, dim);
		}

		<T, U> Output<U> cast(Output<T> value, Class<U> type) {
			DataType dtype = DataType.fromClass(type);
			return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().<U>output(0);
		}

		Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
			return g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents).setAttr("channels", channels).build()
					.<UInt8>output(0);
		}

		<T> Output<T> constant(String name, Object value, Class<T> type) {
			try (Tensor<T> t = Tensor.<T>create(value, type)) {
				return g.opBuilder("Const", name).setAttr("dtype", DataType.fromClass(type)).setAttr("value", t).build()
						.<T>output(0);
			}
		}

		Output<String> constant(String name, byte[] value) {
			return this.constant(name, value, String.class);
		}

		Output<Integer> constant(String name, int value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Integer> constant(String name, int[] value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Float> constant(String name, float value) {
			return this.constant(name, value, Float.class);
		}

		private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}

		private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}
	}
}