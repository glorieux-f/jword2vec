package com.github.oeuvres.jword2vec;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

/**
 * Represents the Word2Vec model, containing vectors for each word
 *
 * @see {@link #forSearch()}
 */
public class VecModel
{
    /** For slicing big MappedByteBuffer https://blog.vanillajava.blog/2011/12/using-memory-mapped-file-for-huge.html */
    static private final long ONE_GB = 1024 * 1024 * 1024;
    /** To get wordId by word */
    protected final Map<String, Integer> word4id;
    /** To get word by wordId */
    protected final String[] vocab;
    /** Size of vectors */
    protected final int layerSize;
    /** File mapped vectors */
    final FloatBuffer vectors;
    
    public boolean contains(String word) {
        return word4id.containsKey(word);
    }

    public Integer wordId(String word) {
        return word4id.get(word);
    }

    VecModel(final String[] vocab, int layerSize, FloatBuffer vectors)
    {
        this.vocab = vocab;
        this.layerSize = layerSize;
        this.vectors = vectors;
        word4id = new HashMap<String, Integer>();
        for (int i = 0; i < vocab.length; i++) {
            word4id.put(vocab[i], i);
        }
    }

    VecModel(final String[] vocab, int layerSize, float[] vectors)
    {
        this(vocab, layerSize, FloatBuffer.wrap(vectors));
    }


    /** @return Layer size */
    public int layerSize()
    {
        return layerSize;
    }

    /** @return {@link VecSearch} for searching */
    public VecSearch forSearch()
    {
        return new VecSearch(this);
    }


    /**
     * Forwards to {@link #fromBinFile(File, ByteOrder, ProfilingTimer)} with the
     * default ByteOrder.LITTLE_ENDIAN and no ProfilingTimer
     */
    public static VecModel fromBinFile(File file) throws IOException
    {
        return fromBinFile(file, ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * @return {@link VecModel} created from the binary representation output
     *         by the open source C version of word2vec using the given byte order.
     */
    public static VecModel fromBinFile(File file, ByteOrder byteOrder) throws IOException
    {
    
        try (final FileInputStream fis = new FileInputStream(file);) {
            final FileChannel channel = fis.getChannel();
            MappedByteBuffer sourceBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0,
                    Math.min(channel.size(), Integer.MAX_VALUE));
            sourceBuffer.order(byteOrder);
            int bufferCount = 1;
            // Java's NIO only allows memory-mapping up to 2GB. To work around this problem,
            // we re-map
            // every gigabyte. To calculate offsets correctly, we have to keep track how
            // many gigabytes
            // we've already skipped. That's what this is for.
    
            StringBuilder sb = new StringBuilder();
            char c = (char) sourceBuffer.get();
            while (c != '\n') {
                sb.append(c);
                c = (char) sourceBuffer.get();
            }
            String firstLine = sb.toString();
            int index = firstLine.indexOf(' ');
            if (index == -1) {
                throw new IndexOutOfBoundsException(
                    String.format(
                        "Expected a space in the first line of file %s: “%s”",
                        file.getAbsolutePath(),
                        firstLine
                    )
                );
            }
    
            final int vocabSize = Integer.parseInt(firstLine.substring(0, index));
            final int layerSize = Integer.parseInt(firstLine.substring(index + 1));
    
            String[] vocab = new String[vocabSize];
            FloatBuffer vectors = ByteBuffer.allocateDirect(vocabSize * layerSize * 4).asFloatBuffer();
    
            final float[] sourceVec = new float[layerSize];
            // https://github.com/medallia/Word2VecJava/issues/44
            // bytes instead of chars
            byte[] buff = new byte[1024];
            for (int lineno = 0; lineno < vocabSize; lineno++) {
                // read vocab
                int bpos = 0;
                byte b = sourceBuffer.get();
                while (b != ' ') {
                    // ignore newlines in front of words (some binary files have newline,
                    // some don't)
                    if (b != '\n') {
                        buff[bpos++] = b;
                    }
                    b = sourceBuffer.get();
                }
                vocab[lineno] = new String(buff, 0, bpos, "UTF-8");
                vectors.put(lineno * layerSize, sourceBuffer.asFloatBuffer(), 0, layerSize);
                sourceBuffer.position(sourceBuffer.position() + 4 * layerSize);
    
    
                // remap file
                if (sourceBuffer.position() > ONE_GB) {
                    final int newPosition = (int) (sourceBuffer.position() - ONE_GB);
                    final long size = Math.min(channel.size() - ONE_GB * bufferCount, Integer.MAX_VALUE);
                    sourceBuffer = channel.map(FileChannel.MapMode.READ_ONLY, ONE_GB * bufferCount, size);
                    sourceBuffer.order(byteOrder);
                    sourceBuffer.position(newPosition);
                    bufferCount += 1;
                }
            }
    
            return new VecModel(
                vocab, 
                layerSize, 
                vectors
            );
        }
    }

    /**
     * @return {@link VecModel} read from a file in the text output format of
     *         the Word2Vec C open source project.
     */
    public static VecModel fromTextFile(File file) throws IOException
    {
        List<String> lines = Files.readAllLines(file.toPath());
        return fromTextFile(file.getAbsolutePath(), lines);
    }

    /**
     * Not tested
     * 
     * @return {@link VecModel} from the lines of the file in the text output
     *         format of the Word2Vec C open source project.
     */
    @VisibleForTesting
    static VecModel fromTextFile(String filename, List<String> lines) throws IOException
    {
        int vocabSize = Integer.parseInt(lines.get(0).split(" ")[0]);
        if (vocabSize != lines.size() - 1) {
            throw new IllegalArgumentException(String.format("%s, vobabSize=%d according to line 0, but %d vector line found", filename, vocabSize, lines.size() - 1));
        }
        int layerSize = Integer.parseInt(lines.get(0).split(" ")[1]);
        String[] vocab = new String[vocabSize];
        float[] vectors = new float[vocabSize * layerSize];
        for (int wordId = 0; wordId < vocabSize; wordId++) {
            String[] values = lines.get(wordId + 1).split(" ");
            if (layerSize != values.length - 1) {
                throw new IllegalArgumentException(String.format("%s#%d, layerSize=%d according to line 0, but %d values fond in vector", filename, wordId + 1, layerSize, values.length - 1));
            }
            vocab[wordId] = values[0];
            final int wordIndex = wordId * layerSize;
            for (int node = 0; node < layerSize; node++) {
                vectors[wordIndex + node] = Float.parseFloat(values[node + 1]);
            }
        }
        
        return new VecModel(
            vocab, 
            layerSize,
            vectors
        );
    }

    /**
     * Saves the model as a bin file that's compatible with the C version of
     * Word2Vec
     */
    public void toBinFile(final OutputStream out) throws IOException
    {
        // could be optimize
        final Charset cs = Charset.forName("UTF-8");
        final String header = String.format("%d %d\n", vocab.length, layerSize);
        out.write(header.getBytes(cs));
    
        final float[] vector = new float[layerSize];
        final ByteBuffer buffer = ByteBuffer.allocate(4 * layerSize);
        buffer.order(ByteOrder.LITTLE_ENDIAN); // The C version uses this byte order.
        for (int wordId = 0; wordId < vocab.length; ++wordId) {
            out.write(String.format("%s ", vocab[wordId]).getBytes(cs));
    
            vectors.position(wordId * layerSize);
            vectors.get(vector);
            buffer.clear();
            for (int j = 0; j < layerSize; ++j)
                buffer.putFloat((float) vector[j]);
            out.write(buffer.array());
    
            out.write('\n');
        }
    
        out.flush();
    }

    /** @return {@link VecTrainerBuilder} for training a model */
    public static VecTrainerBuilder trainer()
    {
        return new VecTrainerBuilder();
    }
}
