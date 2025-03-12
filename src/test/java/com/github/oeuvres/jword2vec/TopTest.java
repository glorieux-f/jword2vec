package com.github.oeuvres.jword2vec;

import static org.junit.Assert.*;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

import com.github.oeuvres.jword2vec.util.Edge;
import com.github.oeuvres.jword2vec.util.Top;

public class TopTest
{
    
    @Test
    public void fill() throws InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException
    {
        Top<Edge> top = new Top<Edge>(Edge.class, 5);
        List<Integer> series = new ArrayList<>();
        for (int i = -5; i <= 5; i++) {
            series.add(i);
        }
        Collections.shuffle(series);
        System.out.println(series);
        for (int score: series) {
            if (!top.isInsertable(score)) continue;
            top.insert(score).sourceLabel("De").targetId(score);
            // System.out.println(score + " " + Arrays.toString(top.data));
        }
        System.out.println(top);
    }
}
