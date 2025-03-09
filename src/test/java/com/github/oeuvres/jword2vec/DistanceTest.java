package com.github.oeuvres.jword2vec;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

import com.github.oeuvres.jword2vec.Searcher.Match;
import com.github.oeuvres.jword2vec.Searcher.UnknownWordException;

public class DistanceTest
{

    static void distance() throws IOException, UnknownWordException
    {

        File modelFile = new File(DistanceTest.class.getResource("rougemont.bin").getFile()); 
        Word2VecModel model = Word2VecModel.fromBinFile(modelFile);
        Searcher searcher =  model.forSearch();
        
        Scanner input = new Scanner(System.in);
        while (true) {
            System.out.print("Proposez un mot : ");
            final String word = input.nextLine();
            if (word.isBlank()) break;
            if (!searcher.contains(word)) {
                System.out.println("Mots absent");
                continue;
            }
            final long timeStart = System.nanoTime();
            List<Match> matches = searcher.getMatches(word, 20);
            System.out.println(((System.nanoTime() - timeStart) / 1000000) + " ms");
            for (final Match match: matches) {
                System.out.println(match.match() + "\t" + match.distance());
            }
        }
        input.close();
    }

    public static void main(String[] args) throws Exception
    {
        distance();
    }

}
