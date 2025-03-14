package com.github.oeuvres.jword2vec;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.ref.Cleaner;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



/**
 * Represents the Word2Vec model, containing vectors for each word
 *
 * @see {@link #forSearch()}
 */
public class VecModel
{
    /** Size of a float number in bytes */
    static final private int FLOAT_BYTES = 4;
    /** Size of a double number in bytes */
    static final private int DOUBLE_BYTES = 8;
    /** Size of a point number in bytes, maybe 8 for double or 4 for float */
    static final private int POINT_BYTES = DOUBLE_BYTES;
    /** For slicing big MappedByteBuffer https://blog.vanillajava.blog/2011/12/using-memory-mapped-file-for-huge.html */
    static private final long ONE_GB = 1024 * 1024 * 1024;
    /** To get wordId by word */
    protected final Map<String, Integer> word4id;
    /** To get word by wordId */
    protected final String[] vocab;
    /** Size of vectors */
    protected final int layerSize;
    /** File mapped vectors */
    final DoubleBuffer vectors;
    
    public boolean contains(String word) {
        return word4id.containsKey(word);
    }

    public Integer wordId(String word) {
        return word4id.get(word);
    }

    VecModel(final String[] vocab, int layerSize, DoubleBuffer vectors)
    {
        this.vocab = vocab;
        this.layerSize = layerSize;
        this.vectors = vectors;
        word4id = new HashMap<String, Integer>();
        for (int i = 0; i < vocab.length; i++) {
            word4id.put(vocab[i], i);
        }
    }

    VecModel(final String[] vocab, int layerSize, double[] vectors)
    {
        this(vocab, layerSize, DoubleBuffer.wrap(vectors));
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
     * @throws SecurityException 
     * @throws NoSuchMethodException 
     * @throws InvocationTargetException 
     * @throws IllegalAccessException 
     */
    public static VecModel fromBinFile(File file, ByteOrder byteOrder) throws IOException
    {
        String[] vocab;
        DoubleBuffer vectors;
        final int layerSize;
        MappedByteBuffer binBuffer = null; // release to be forces
        try (
            final FileInputStream fis = new FileInputStream(file);
            final FileChannel channel = fis.getChannel();
        ){
            binBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0,
                    Math.min(channel.size(), Integer.MAX_VALUE));
            binBuffer.order(byteOrder);
            int bufferCount = 1;
            // Java's NIO only allows memory-mapping up to 2GB. To work around this problem,
            // we re-map
            // every gigabyte. To calculate offsets correctly, we have to keep track how
            // many gigabytes
            // we've already skipped. That's what this is for.
    
            StringBuilder sb = new StringBuilder();
            char c = (char) binBuffer.get();
            while (c != '\n') {
                sb.append(c);
                c = (char) binBuffer.get();
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
            layerSize = Integer.parseInt(firstLine.substring(index + 1));
    
            vocab = new String[vocabSize];
            vectors = ByteBuffer.allocateDirect(vocabSize * layerSize * POINT_BYTES).asDoubleBuffer();
    
            final float[] binVec = new float[layerSize];
            // https://github.com/medallia/Word2VecJava/issues/44
            // bytes instead of chars
            byte[] buff = new byte[1024];
            for (int lineno = 0; lineno < vocabSize; lineno++) {
                // read vocab
                int bpos = 0;
                byte b = binBuffer.get();
                while (b != ' ') {
                    // ignore newlines in front of words (some binary files have newline,
                    // some don't)
                    if (b != '\n') {
                        buff[bpos++] = b;
                    }
                    b = binBuffer.get();
                }
                vocab[lineno] = new String(buff, 0, bpos, "UTF-8");
                // read float vector from model, and load it as double for memory
                final FloatBuffer floatBuffer = binBuffer.asFloatBuffer();
                floatBuffer.get(binVec);
                for (int i = 0; i < binVec.length; ++i) {
                    vectors.put(lineno * layerSize + i, binVec[i]);
                }
                
                
                binBuffer.position(binBuffer.position() + FLOAT_BYTES * layerSize);
    
    
                // remap file
                if (binBuffer.position() > ONE_GB) {
                    final int newPosition = (int) (binBuffer.position() - ONE_GB);
                    final long size = Math.min(channel.size() - ONE_GB * bufferCount, Integer.MAX_VALUE);
                    binBuffer = channel.map(FileChannel.MapMode.READ_ONLY, ONE_GB * bufferCount, size);
                    binBuffer.order(byteOrder);
                    binBuffer.position(newPosition);
                    bufferCount += 1;
                }
            }
            binBuffer.clear();
        }
        // 
        finally {
            if (binBuffer != null) {
                binBuffer.clear();
            }
        }
        return new VecModel(
            vocab, 
            layerSize, 
            vectors
        );
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
    static VecModel fromTextFile(String filename, List<String> lines) throws IOException
    {
        int vocabSize = Integer.parseInt(lines.get(0).split(" ")[0]);
        if (vocabSize != lines.size() - 1) {
            throw new IllegalArgumentException(String.format("%s, vobabSize=%d according to line 0, but %d vector line found", filename, vocabSize, lines.size() - 1));
        }
        int layerSize = Integer.parseInt(lines.get(0).split(" ")[1]);
        String[] vocab = new String[vocabSize];
        double[] vectors = new double[vocabSize * layerSize];
        for (int wordId = 0; wordId < vocabSize; wordId++) {
            String[] values = lines.get(wordId + 1).split(" ");
            if (layerSize != values.length - 1) {
                throw new IllegalArgumentException(String.format("%s#%d, layerSize=%d according to line 0, but %d values fond in vector", filename, wordId + 1, layerSize, values.length - 1));
            }
            vocab[wordId] = values[0];
            final int wordIndex = wordId * layerSize;
            for (int node = 0; node < layerSize; node++) {
                vectors[wordIndex + node] = Double.parseDouble(values[node + 1]);
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
    
        final double[] vector = new double[layerSize];
        final ByteBuffer binBuffer = ByteBuffer.allocate(FLOAT_BYTES * layerSize);
        binBuffer.order(ByteOrder.LITTLE_ENDIAN); // The C version uses this byte order.
        for (int wordId = 0; wordId < vocab.length; ++wordId) {
            out.write(String.format("%s ", vocab[wordId]).getBytes(cs));
    
            vectors.position(wordId * layerSize);
            vectors.get(vector);
            
            binBuffer.clear();
            for (int j = 0; j < layerSize; ++j)
                binBuffer.putFloat((float) vector[j]);
            out.write(binBuffer.array());
    
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
