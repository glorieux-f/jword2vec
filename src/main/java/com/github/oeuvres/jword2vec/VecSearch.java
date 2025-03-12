package com.github.oeuvres.jword2vec;

import com.github.oeuvres.jword2vec.util.Edge;
import com.github.oeuvres.jword2vec.util.Top;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.common.primitives.Doubles;

import java.nio.DoubleBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Provides search functionality */
public class VecSearch
{
    private final NormalizedWord2VecModel model;

    VecSearch(final NormalizedWord2VecModel model)
    {
        this.model = model;

    }

    VecSearch(final VecModel model)
    {
        this(NormalizedWord2VecModel.fromWord2VecModel(model));
    }

    /** @return true if a word is inside the model's vocabulary. */
    public boolean contains(String word)
    {
        return model.contains(word);
    }

    /**
     * 
     * @param words
     * @param limit
     * @return
     * @throws UnknownWordException
     */
    public Edge[] sims(final String[] words, final int limit) throws UnknownWordException
    {
        double[][] vectors = new double[words.length][];
        for (int v = 0; v < words.length; v++) {
            vectors[v] = vector(words[v]);
        }
        double[] mean = mean(vectors);
        Edge[] edges = sims(mean, limit);
        return edges;
    }

    /**
     * Loop on all vectors of a model to find the closest
     * to the given one.
     * @param vec
     * @param limit
     * @return
     */
    public Edge[] sims(final double[] vec, int limit) 
    {
        // the top collector
        Top<Edge> top = new Top<>(Edge.class, limit);
        if (vec == null) {
            throw new IllegalArgumentException("Reference vector is required");
        }
        if (vec.length != model.layerSize) {
            throw new IllegalArgumentException(String.format("vec.length=%d != model.layerSize=%d, bad vector", vec.length, model.layerSize));
        }
        // ensure source vector has no NaN
        for(int node = 0; node < model.layerSize; node ++) {
            if (Double.isNaN(vec[node])) vec[node] = 0;
        }
        final DoubleBuffer vectors = model.vectors.duplicate();
        for (int wordId = 0; wordId < model.vocab.length; wordId++) {
            // calculate cosine distance
            double score = 0;
            for(int node = 0; node < model.layerSize; node ++) {
                final double d2 = vectors.get();
                if (Double.isNaN(d2)) continue;
                score += vec[node] * d2;
            }
            if (!top.isInsertable(score)) continue;
            top.insert(score).targetId(wordId).score(score);
        }
        Edge[] edges = top.toArray();
        for (int i = 0; i < edges.length; i++) {
            edges[i].targetLabel(model.vocab[edges[i].targetId()]);
        }
        return edges;
    }

    private double[] vector(final String word) throws UnknownWordException
    {
        final Integer wordId = model.wordId(word);
        if (wordId == null) {
            throw new UnknownWordException(word);
        }
        int position = wordId * model.layerSize;
        // buffers' position, limit, and mark values will beindependent.
        final DoubleBuffer vectors = model.vectors.duplicate();
        double[] result = new double[model.layerSize];
        vectors.position(position);
        vectors.get(result);
        return result;
    }

    private double cosine(double[] vec1, double[] vec2)
    {
        double d = 0;
        for (int a = 0; a < model.layerSize; a++) {
            // NaN bug in loading
            if (Double.isNaN(vec1[a]) || Double.isNaN(vec2[a]))
                continue;
            d += vec2[a] * vec1[a];
        }
        return d;
    }

    /** @return Vector difference from v1 to v2 */
    private double[] difference(double[] v1, double[] v2)
    {
        double[] diff = new double[model.layerSize];
        for (int i = 0; i < model.layerSize; i++)
            diff[i] = v1[i] - v2[i];
        return diff;
    }

    /**
     * @return Vector mean
     */
    private double[] mean(double[][] vectors)
    {
        double[] mean = new double[model.layerSize];
        for (int p = 0; p < model.layerSize; p++) {
            double sum = 0;
            int card = 0;
            for (int v = 0; v < vectors.length; v++) {
                // skip NaN (infinity ?)
                if (Double.isNaN(vectors[v][p]))
                    continue;
                sum += vectors[v][p];
                card++;
            }
            mean[p] = sum / card;
        }
        return mean;
    }

    /**
     * Exception when a word is unknown to the {@link VecModel}'s vocabulary
     */
    @SuppressWarnings("serial")
    public static class UnknownWordException extends Exception
    {
        UnknownWordException(String word)
        {
            super(String.format("Unknown search word “%s”", word));
        }
    }
}
