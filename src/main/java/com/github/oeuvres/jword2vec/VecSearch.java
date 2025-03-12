package com.github.oeuvres.jword2vec;

import com.github.oeuvres.alix.util.Edge;
import com.github.oeuvres.alix.util.Top;

import java.nio.FloatBuffer;

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
        float[][] vectors = new float[words.length][];
        for (int v = 0; v < words.length; v++) {
            vectors[v] = vector(words[v]);
        }
        float[] mean = mean(vectors);
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
    public Edge[] sims(final float[] vec, int limit) 
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
            if (Float.isNaN(vec[node])) vec[node] = 0;
        }
        final FloatBuffer vectors = model.vectors.duplicate();
        for (int wordId = 0; wordId < model.vocab.length; wordId++) {
            // calculate cosine distance
            double score = 0;
            for(int node = 0; node < model.layerSize; node ++) {
                final float d2 = vectors.get();
                if (Float.isNaN(d2)) continue;
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

    private float[] vector(final String word) throws UnknownWordException
    {
        final Integer wordId = model.wordId(word);
        if (wordId == null) {
            throw new UnknownWordException(word);
        }
        int position = wordId * model.layerSize;
        // buffers' position, limit, and mark values will beindependent.
        final FloatBuffer vectors = model.vectors.duplicate();
        float[] result = new float[model.layerSize];
        vectors.position(position);
        vectors.get(result);
        return result;
    }

    private double cosine(float[] vec1, float[] vec2)
    {
        double d = 0;
        for (int a = 0; a < model.layerSize; a++) {
            // NaN bug in loading
            if (Float.isNaN(vec1[a]) || Float.isNaN(vec2[a]))
                continue;
            d += vec2[a] * vec1[a];
        }
        return d;
    }

    /** @return Vector difference from v1 to v2 */
    private float[] difference(float[] v1, float[] v2)
    {
        float[] diff = new float[model.layerSize];
        for (int i = 0; i < model.layerSize; i++)
            diff[i] = v1[i] - v2[i];
        return diff;
    }

    /**
     * @return Vector mean
     */
    private float[] mean(float[][] vectors)
    {
        float[] mean = new float[model.layerSize];
        for (int p = 0; p < model.layerSize; p++) {
            double sum = 0;
            int card = 0;
            for (int v = 0; v < vectors.length; v++) {
                // skip NaN (infinity ?)
                if (Float.isNaN(vectors[v][p]))
                    continue;
                sum += vectors[v][p];
                card++;
            }
            mean[p] = (float)(sum / card);
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
