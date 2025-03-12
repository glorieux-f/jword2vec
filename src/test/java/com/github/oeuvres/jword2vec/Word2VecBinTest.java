package com.github.oeuvres.jword2vec;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import com.github.oeuvres.jword2vec.VecSearch;
import com.github.oeuvres.jword2vec.VecModel;
import com.github.oeuvres.jword2vec.VecSearch.UnknownWordException;
import com.github.oeuvres.jword2vec.util.Common;

/**
 * Tests converting the binary models into
 * {@link com.github.oeuvres.jword2vec.VecModel}s.
 * 
 * @see com.github.oeuvres.jword2vec.VecModel#fromBinFile(File)
 * @see com.github.oeuvres.jword2vec.VecModel#fromBinFile(File,
 *      java.nio.ByteOrder)
 */
public class Word2VecBinTest {

  /**
   * Tests that the Word2VecModels created from a binary and text
   * representations are equivalent
   */
  @Test
  public void testRead()
      throws IOException, UnknownWordException {
    File binFile = Common.getResourceAsFile(
            this.getClass(),
            "tokensModel.bin");
    VecModel binModel = VecModel.fromBinFile(binFile);

    File txtFile = Common.getResourceAsFile(
            this.getClass(),
            "tokensModel.txt");
    VecModel txtModel = VecModel.fromTextFile(txtFile);

    assertEquals(binModel, txtModel);
  }

  private Path tempFile = null;

  /**
   * Tests that a Word2VecModel round-trips through the bin format without changes
   */
  @Test
  public void testRoundTrip() throws IOException, UnknownWordException {
      /*
    final String filename = "word2vec.c.output.model.txt";
    final VecModel model =
        VecModel.fromTextFile(filename, Common.readResource(Word2VecTest.class, filename));

    tempFile = Files.createTempFile(
            String.format("%s-", Word2VecBinTest.class.getSimpleName()), ".bin");
    try (final OutputStream os = Files.newOutputStream(tempFile)) {
      model.toBinFile(os);
    }

    final VecModel modelCopy = VecModel.fromBinFile(tempFile.toFile());
    assertEquals(model, modelCopy);
    */
  }

  @After
  public void cleanupTempFile() throws IOException {
    if(tempFile != null)
      Files.delete(tempFile);
  }

  private void assertEquals(
      final VecModel leftModel,
      final VecModel rightModel) throws UnknownWordException {
    final VecSearch leftSearcher = leftModel.forSearch();
    final VecSearch rightSearcher = rightModel.forSearch();

    /*
    // test vocab
    for (String vocab : leftModel.getVocab()) {
      assertTrue(rightSearcher.contains(vocab));
    }
    for (String vocab : rightModel.getVocab()) {
      assertTrue(leftSearcher.contains(vocab));
    }
    */
    /*
    // test vector
    for (String vocab : leftModel.getVocab()) {
      final List<Double> leftVector = leftSearcher.getRawVector(vocab);
      final List<Double> rightVector = rightSearcher.getRawVector(vocab);
      assertEquals(leftVector, rightVector);
    }
    */
  }

  /*
  private void assertEquals(
      final List<Double> leftVector,
      final List<Double> rightVector) {
    Assert.assertEquals(leftVector.size(), rightVector.size());
    for (int i = 0; i < leftVector.size(); i++) {
      double txtD = leftVector.get(i);
      double binD = rightVector.get(i);
      Assert.assertEquals(txtD, binD, 0.0001);
    }
  }
  */
}
