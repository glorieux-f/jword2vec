package com.github.oeuvres.jword2vec;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import com.github.oeuvres.alix.util.Edge;
import com.github.oeuvres.jword2vec.VecSearch.UnknownWordException;

public class DistanceTest
{

    static void distance() throws IOException, UnknownWordException
    {

        // File modelFile = new File(DistanceTest.class.getResource("rougemont.bin").getFile());
        // File modelFile = new File("D:/code/ddr_lab/src/main/webapp/rougemont.bin");
        File modelFile = new File("D:/code/word2vec/piaget.bin");
        VecModel model = VecModel.fromBinFile(modelFile);
        VecSearch searcher =  model.forSearch();
        
        Scanner input = new Scanner(System.in);
        while (true) {
            System.out.print("Proposez un ou plusieurs mots : ");
            final String line = input.nextLine();
            if (line.isBlank()) break;
            String[] words = line.split("\s+");
            boolean redo = false;
            for (String word: words) {
                if (searcher.contains(word)) continue;
                System.out.println("Mot absent : " + word);
                redo = true;
            }
            if (redo) continue;
            final long timeStart = System.nanoTime();
            Edge[] edges = searcher.sims(words, 20);
            System.out.println(((System.nanoTime() - timeStart) / 1000000) + " ms");
            for (final Edge edge: edges) {
                System.out.println(edge.targetLabel() + "\t" + edge.score());
            }
        }
        input.close();
    }

    public static void main(String[] args) throws Exception
    {
        distance();
    }

}
