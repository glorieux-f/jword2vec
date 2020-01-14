import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Searcher.Match;
import com.medallia.word2vec.Searcher.UnknownWordException;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.util.AutoLog;
import org.apache.commons.logging.Log;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Example usages of {@link Word2VecModel}
 */
public class Word2VecExamples {
    private static final Log LOG = AutoLog.getLog();

    /**
     * Runs the example
     */
    public static void main(String[] args) throws IOException, UnknownWordException {
        System.out.println("====start====");
        loadModel();
    }

    /**
     * Trains a model and allows user to find similar words
     * demo-word.sh example from the open source C implementation
     */

    /**
     * Loads a model and allows user to find similar words
     */
    public static void loadModel() throws IOException {
        final Word2VecModel model;
        model = Word2VecModel.fromBinFile(new File("news_embedding_vector.w2v.bin"));
        Searcher searcher = model.forSearch();
        try {
            List<Match> simItems = searcher.getMatches("5083054550058670598", 0);
            System.out.println(simItems);
            for (int i = 0; i < simItems.size(); i++) {
                double distance = simItems.get(i).distance();
                String item = simItems.get(i).match();
                if ("5083054550058670598".equals(item) || distance < 0.7) {
                    continue;
                }
                System.out.println(item + ":" + distance);

            }

        } catch (UnknownWordException e) {
            System.out.println("OOV");
        }
    }
}
