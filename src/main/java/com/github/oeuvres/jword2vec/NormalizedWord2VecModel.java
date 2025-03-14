package com.github.oeuvres.jword2vec;

import java.io.File;
import java.io.IOException;
import java.nio.DoubleBuffer;

/**
 * Represents a word2vec model where all the vectors are normalized to unit
 * length.
 */
public class NormalizedWord2VecModel extends VecModel
{
    private NormalizedWord2VecModel(String[] vocab, int layerSize, final DoubleBuffer vectors)
    {
        super(vocab, layerSize, vectors);
        normalize();
    }

    private NormalizedWord2VecModel(String[] vocab, int layerSize, double[] vectors)
    {
        super(vocab, layerSize, vectors);
        normalize();
    }

    public static NormalizedWord2VecModel fromWord2VecModel(VecModel model)
    {
        return new NormalizedWord2VecModel(model.vocab, model.layerSize, model.vectors.duplicate());
    }

    public static NormalizedWord2VecModel fromBinFile(final File file) throws IOException
    {
        return fromWord2VecModel(VecModel.fromBinFile(file));
    }

    /** Normalizes the vectors in this model */
    private void normalize()
    {
        for (int i = 0; i < vocab.length; ++i) {
            double len = 0;
            for (int j = i * layerSize; j < (i + 1) * layerSize; ++j)
                len += vectors.get(j) * vectors.get(j);
            len = (double) Math.sqrt(len);

            for (int j = i * layerSize; j < (i + 1) * layerSize; ++j)
                vectors.put(j, vectors.get(j) / len);
        }
    }
}
