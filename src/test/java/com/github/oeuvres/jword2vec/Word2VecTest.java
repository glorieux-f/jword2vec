package com.github.oeuvres.jword2vec;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.List;

import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import com.github.oeuvres.jword2vec.Searcher.Match;
import com.github.oeuvres.jword2vec.Searcher.UnknownWordException;
import com.github.oeuvres.jword2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.github.oeuvres.jword2vec.neuralnetwork.NeuralNetworkType;
import com.github.oeuvres.jword2vec.util.Common;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

/**
 * Tests for {@link Word2VecModel} and related classes.
 * <p>
 * Note that the implementation is expected to be deterministic if numThreads is
 * set to 1
 */
public class Word2VecTest {
	@Rule
	public ExpectedException expected = ExpectedException.none();

	/** Clean up after a test run */
	@After
	public void after() {
		// Unset the interrupted flag to avoid polluting other tests
		Thread.interrupted();
	}

	/** Test that we can interrupt the huffman encoding process */
	@Test
	public void testInterruptHuffman() throws IOException, InterruptedException {
		expected.expect(InterruptedException.class);
		trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.setNumIterations(15)
			.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.CREATE_HUFFMAN_ENCODING)
							Thread.currentThread().interrupt();
						else if (stage == Stage.TRAIN_NEURAL_NETWORK)
							fail("Should not have reached this stage");
					}
				})
			.train(testData());
	}

	/** Test that we can interrupt the neural network training process */
	@Test
	public void testInterruptNeuralNetworkTraining() throws InterruptedException, IOException {
		expected.expect(InterruptedException.class);
		trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.setNumIterations(15)
			.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.TRAIN_NEURAL_NETWORK)
							Thread.currentThread().interrupt();
					}
				})
			.train(testData());
	}

  /**
   * Test the search results are deterministic Note the actual values may not
   * make sense since the model we train isn't tuned
   */
	@Test
	public void testSearch() throws InterruptedException, IOException, UnknownWordException {
		Word2VecModel model = trainer()
			.type(NeuralNetworkType.SKIP_GRAM)
			.train(testData());

		List<Match> matches = model.forSearch().getMatches("anarchism", 5);

		assertEquals(
				ImmutableList.of("anarchism", "feminism", "trouble", "left", "capitalism"),
				Lists.transform(matches, Match.TO_WORD)
			);
	}

  /**
   * Test that the model can retrieve words by a vector.
   */
  @Test
    public void testGetWordByVector() throws InterruptedException, IOException, UnknownWordException {
        Word2VecModel model = trainer()
            .type(NeuralNetworkType.SKIP_GRAM)
            .train(testData());

        // This vector defines the word "anarchism" in the given model.
        double[] vectors = new double[] { 0.11410251703652753, 0.271180824514185, 0.03748515103121994, 0.20888126888511183, 0.009713531343874777, 0.4769425625416319, 0.1431890482445165, -0.1917578875330224, -0.33532561802423366,
            -0.08794543238607992, 0.20404593606213406, 0.26170074241479385, 0.10020961212561065, 0.11400571893146201, -0.07846426915175395, -0.19404092647187385, 0.13381991303455204, -4.6749635342694615E-4, -0.0820905789076496,
            -0.30157145455251866, 0.3652037905836543, -0.16466827556950117, -0.012965932276668056, 0.09896568721267748, -0.01925755122093615 };

        List<Match> matches = model.forSearch().getMatches(vectors, 5);

        assertEquals(
                ImmutableList.of("anarchism", "feminism", "trouble", "left", "capitalism"),
                Lists.transform(matches, Match.TO_WORD)
            );
    }
  
  /**
   * Test that the model can retrieve words by a vector.
   */
  @Test
    public void testGetWordByNotExistantVector() throws InterruptedException, IOException, UnknownWordException {
        Word2VecModel model = trainer()
            .type(NeuralNetworkType.SKIP_GRAM)
            .train(testData());

        double[] vectors = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0 };

        List<Match> matches = model.forSearch().getMatches(vectors, 5);

        assertEquals(
                ImmutableList.of("the", "of", "and", "in", "a"),
                Lists.transform(matches, Match.TO_WORD)
            );
    }

	/** Test reading Word2Vec C version txt output format into this library */
	@Test
	public void testTxtModelRead() throws IOException, UnknownWordException {
		String filename = "word2vec.c.output.model.txt";
		Word2VecModel word2VecModel = Word2VecModel.fromTextFile(filename, Common.readResource(Word2VecTest.class, filename));
    assertEquals(0.9927725293757652, word2VecModel.forSearch().cosineDistance("three", "five"), 1e-5);
	}

	/** @return {@link Word2VecTrainer} which by default uses all of the supported features */
	@VisibleForTesting
	public static Word2VecTrainerBuilder trainer() {
		return Word2VecModel.trainer()
			.setMinVocabFrequency(6)
			.useNumThreads(1)
			.setWindowSize(8)
			.type(NeuralNetworkType.CBOW)
			.useHierarchicalSoftmax()
			.setLayerSize(25)
			.setDownSamplingRate(1e-3)
			.setNumIterations(1);
	}

	/** @return raw test dataset. The tokens are separated by newlines. */
	@VisibleForTesting
	public static Iterable<List<String>> testData() throws IOException {
		List<String> lines = Common.readResource(Word2VecTest.class, "word2vec.short.txt");
		Iterable<List<String>> partitioned = Iterables.partition(lines, 1000);
		return partitioned;
	}

}
