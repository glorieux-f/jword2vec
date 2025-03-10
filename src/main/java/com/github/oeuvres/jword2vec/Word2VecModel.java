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
import java.util.List;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

/**
 * Represents the Word2Vec model, containing vectors for each word
 * <p/>
 * Instances of this class are obtained via:
 * <ul>
 * <li>{@link #trainer()}
 * <li>{@link #fromThrift(Word2VecModelThrift)}
 * </ul>
 *
 * @see {@link #forSearch()}
 */
public class Word2VecModel
{
    final List<String> vocab;
    final int layerSize;
    final DoubleBuffer vectors;
    private final static long ONE_GB = 1024 * 1024 * 1024;

    Word2VecModel(Iterable<String> vocab, int layerSize, DoubleBuffer vectors)
    {
        this.vocab = ImmutableList.copyOf(vocab);
        this.layerSize = layerSize;
        this.vectors = vectors;
    }

    Word2VecModel(Iterable<String> vocab, int layerSize, double[] vectors)
    {
        this(vocab, layerSize, DoubleBuffer.wrap(vectors));
    }

    /** @return Vocabulary */
    public Iterable<String> getVocab()
    {
        return vocab;
    }

    /** @return Layer size */
    public int getLayerSize()
    {
        return layerSize;
    }

    /** @return {@link Searcher} for searching */
    public Searcher forSearch()
    {
        return new SearcherImpl(this);
    }


    /**
     * @return {@link Word2VecModel} read from a file in the text output format of
     *         the Word2Vec C open source project.
     */
    public static Word2VecModel fromTextFile(File file) throws IOException
    {
        List<String> lines = Files.readAllLines(file.toPath());
        return fromTextFile(file.getAbsolutePath(), lines);
    }

    /**
     * Forwards to {@link #fromBinFile(File, ByteOrder, ProfilingTimer)} with the
     * default ByteOrder.LITTLE_ENDIAN and no ProfilingTimer
     */
    public static Word2VecModel fromBinFile(File file) throws IOException
    {
        return fromBinFile(file, ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * @return {@link Word2VecModel} created from the binary representation output
     *         by the open source C version of word2vec using the given byte order.
     */
    public static Word2VecModel fromBinFile(File file, ByteOrder byteOrder) throws IOException
    {

        try (final FileInputStream fis = new FileInputStream(file);) {
            final FileChannel channel = fis.getChannel();
            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0,
                    Math.min(channel.size(), Integer.MAX_VALUE));
            buffer.order(byteOrder);
            int bufferCount = 1;
            // Java's NIO only allows memory-mapping up to 2GB. To work around this problem,
            // we re-map
            // every gigabyte. To calculate offsets correctly, we have to keep track how
            // many gigabytes
            // we've already skipped. That's what this is for.

            StringBuilder sb = new StringBuilder();
            char c = (char) buffer.get();
            while (c != '\n') {
                sb.append(c);
                c = (char) buffer.get();
            }
            String firstLine = sb.toString();
            int index = firstLine.indexOf(' ');
            Preconditions.checkState(index != -1, "Expected a space in the first line of file '%s': '%s'",
                    file.getAbsolutePath(), firstLine);

            final int vocabSize = Integer.parseInt(firstLine.substring(0, index));
            final int layerSize = Integer.parseInt(firstLine.substring(index + 1));

            List<String> vocabs = new ArrayList<String>(vocabSize);
            DoubleBuffer vectors = ByteBuffer.allocateDirect(vocabSize * layerSize * 8).asDoubleBuffer();

            long lastLogMessage = System.currentTimeMillis();
            final float[] floats = new float[layerSize];
            // https://github.com/medallia/Word2VecJava/issues/44
            // bytes instead of chars
            byte[] buff = new byte[1024];
            for (int lineno = 0; lineno < vocabSize; lineno++) {
                // read vocab
                int bpos = 0;
                byte b = buffer.get();
                while (b != ' ') {
                    // ignore newlines in front of words (some binary files have newline,
                    // some don't)
                    if (b != '\n') {
                        buff[bpos++] = b;
                    }
                    b = buffer.get();
                }
                vocabs.add(new String(buff, 0, bpos, "UTF-8"));
                // read vector
                final FloatBuffer floatBuffer = buffer.asFloatBuffer();
                floatBuffer.get(floats);
                for (int i = 0; i < floats.length; ++i) {
                    vectors.put(lineno * layerSize + i, floats[i]);
                }
                buffer.position(buffer.position() + 4 * layerSize);

                // print log
                final long now = System.currentTimeMillis();
                if (now - lastLogMessage > 1000) {
                    final double percentage = ((double) (lineno + 1) / (double) vocabSize) * 100.0;
                    lastLogMessage = now;
                }

                // remap file
                if (buffer.position() > ONE_GB) {
                    final int newPosition = (int) (buffer.position() - ONE_GB);
                    final long size = Math.min(channel.size() - ONE_GB * bufferCount, Integer.MAX_VALUE);
                    buffer = channel.map(FileChannel.MapMode.READ_ONLY, ONE_GB * bufferCount, size);
                    buffer.order(byteOrder);
                    buffer.position(newPosition);
                    bufferCount += 1;
                }
            }

            return new Word2VecModel(vocabs, layerSize, vectors);
        }
    }

    /**
     * Saves the model as a bin file that's compatible with the C version of
     * Word2Vec
     */
    public void toBinFile(final OutputStream out) throws IOException
    {
        final Charset cs = Charset.forName("UTF-8");
        final String header = String.format("%d %d\n", vocab.size(), layerSize);
        out.write(header.getBytes(cs));

        final double[] vector = new double[layerSize];
        final ByteBuffer buffer = ByteBuffer.allocate(4 * layerSize);
        buffer.order(ByteOrder.LITTLE_ENDIAN); // The C version uses this byte order.
        for (int i = 0; i < vocab.size(); ++i) {
            out.write(String.format("%s ", vocab.get(i)).getBytes(cs));

            vectors.position(i * layerSize);
            vectors.get(vector);
            buffer.clear();
            for (int j = 0; j < layerSize; ++j)
                buffer.putFloat((float) vector[j]);
            out.write(buffer.array());

            out.write('\n');
        }

        out.flush();
    }

    /**
     * @return {@link Word2VecModel} from the lines of the file in the text output
     *         format of the Word2Vec C open source project.
     */
    @VisibleForTesting
    static Word2VecModel fromTextFile(String filename, List<String> lines) throws IOException
    {
        List<String> vocab = Lists.newArrayList();
        List<Double> vectors = Lists.newArrayList();
        int vocabSize = Integer.parseInt(lines.get(0).split(" ")[0]);
        int layerSize = Integer.parseInt(lines.get(0).split(" ")[1]);

        Preconditions.checkArgument(vocabSize == lines.size() - 1,
                "For file '%s', vocab size is %s, but there are %s word vectors in the file", filename, vocabSize,
                lines.size() - 1);

        for (int n = 1; n < lines.size(); n++) {
            String[] values = lines.get(n).split(" ");
            vocab.add(values[0]);

            // Sanity check
            Preconditions.checkArgument(layerSize == values.length - 1,
                    "For file '%s', on line %s, layer size is %s, but found %s values in the word vector", filename, n,
                    layerSize, values.length - 1);

            for (int d = 1; d < values.length; d++) {
                vectors.add(Double.parseDouble(values[d]));
            }
        }
        
        return new Word2VecModel(
            vocab,
            layerSize,
            Doubles.toArray(vectors)
        );
    }

    /** @return {@link Word2VecTrainerBuilder} for training a model */
    public static Word2VecTrainerBuilder trainer()
    {
        return new Word2VecTrainerBuilder();
    }
}
