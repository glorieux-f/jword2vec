package com.github.oeuvres.jword2vec;

import com.github.oeuvres.jword2vec.Searcher.Match;
import com.github.oeuvres.jword2vec.Searcher.UnknownWordException;
import com.github.oeuvres.jword2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.github.oeuvres.jword2vec.neuralnetwork.NeuralNetworkType;
import com.github.oeuvres.jword2vec.util.AutoLog;
import com.github.oeuvres.jword2vec.util.Common;
import com.github.oeuvres.jword2vec.util.Format;
import com.github.oeuvres.jword2vec.util.Strings;
import com.github.oeuvres.jword2vec.util.ThriftUtils;
import com.google.common.base.Function;
import com.google.common.collect.Lists;

import org.apache.commons.logging.Log;
import org.apache.thrift.TException;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/** Example usages of {@link Word2VecModel} */
public class Word2VecExamples {
	private static final Log LOG = AutoLog.getLog();
	
	/** Runs the example */
	public static void main(String[] args) throws IOException, TException, UnknownWordException, InterruptedException {
		demoWord();
	}
	
	/** 
	 * Trains a model and allows user to find similar words
	 * demo-word.sh example from the open source C implementation
	 */
	public static void demoWord() throws IOException, TException, InterruptedException, UnknownWordException {
		File f = new File("text8");
		if (!f.exists())
	       	       throw new IllegalStateException("Please download and unzip the text8 example from http://mattmahoney.net/dc/text8.zip");
		List<String> read = Files.readAllLines(f.toPath());
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override
			public List<String> apply(String input) {
				return Arrays.asList(input.split(" "));
			}
		});
		
		Word2VecModel model = Word2VecModel.trainer()
				.setMinVocabFrequency(5)
				.useNumThreads(20)
				.setWindowSize(8)
				.type(NeuralNetworkType.CBOW)
				.setLayerSize(200)
				.useNegativeSamples(25)
				.setDownSamplingRate(1e-4)
				.setNumIterations(5)
				.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
					}
				})
				.train(partitioned);

		// Writes model to a thrift file
		final String content = ThriftUtils.serializeJson(model.toThrift());
		Files.write( Paths.get("text8.model"), content.getBytes("UTF-8"));


		// Alternatively, you can write the model to a bin file that's compatible with the C
		// implementation.
		try(final OutputStream os = Files.newOutputStream(Paths.get("text8.bin"))) {
			model.toBinFile(os);
		}
		
		interact(model.forSearch());
	}
	
	/** Loads a model and allows user to find similar words */
	public static void loadModel() throws IOException, TException, UnknownWordException {
		final Word2VecModel model;
		String json = Files.readString(Path.of("text8.model"));
		model = Word2VecModel.fromThrift(ThriftUtils.deserializeJson(new Word2VecModelThrift(), json));
		interact(model.forSearch());
	}
	
	/** Example using Skip-Gram model */
	public static void skipGram() throws IOException, TException, InterruptedException, UnknownWordException {
		List<String> read = Files.readAllLines(new File("sents.cleaned.word2vec.txt").toPath());
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override
			public List<String> apply(String input) {
				return Arrays.asList(input.split(" "));
			}
		});
		
		Word2VecModel model = Word2VecModel.trainer()
				.setMinVocabFrequency(100)
				.useNumThreads(20)
				.setWindowSize(7)
				.type(NeuralNetworkType.SKIP_GRAM)
				.useHierarchicalSoftmax()
				.setLayerSize(300)
				.useNegativeSamples(0)
				.setDownSamplingRate(1e-3)
				.setNumIterations(5)
				.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
					}
				})
				.train(partitioned);
		
		final String content = ThriftUtils.serializeJson(model.toThrift());
		Files.write( Paths.get("300layer.20threads.5iter.model"), content.getBytes("UTF-8"));
		interact(model.forSearch());
	}
	
	private static void interact(Searcher searcher) throws IOException, UnknownWordException {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String word = br.readLine();
				if (word.equals("EXIT")) {
					break;
				}
				List<Match> matches = searcher.getMatches(word, 20);
				System.out.println(Strings.joinObjects("\n", matches));
			}
		}
	}
}
