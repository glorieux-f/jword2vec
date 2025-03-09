package com.github.oeuvres.jword2vec.util;


import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.io.Serializable;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Simple utilities that in no way deserve their own class.
 */
public class Common {
	/**
	 * @param distance use 1 for our caller, 2 for their caller, etc...
	 * @return the stack trace element from where the calling method was invoked
	 */
	public static StackTraceElement myCaller(int distance) {
		// 0 here, 1 our caller, 2 their caller
		int index = distance + 1;
		try {
			StackTraceElement st[] = new Throwable().getStackTrace();
			// hack: skip synthetic caster methods
			if (st[index].getLineNumber() == 1) return st[index + 1];
			return st[index];
		} catch (Throwable t) {
			return new StackTraceElement("[unknown]","-","-",0);
		}
	}

	/** Serialize the given object into the given stream */
	public static void serialize(Serializable obj, ByteArrayOutputStream bout) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(bout);
			out.writeObject(obj);
			out.close();
		} catch (IOException e) {
			throw new IllegalStateException("Could not serialize " + obj, e);
		}
	}

	/** Read the Reader line for line and return the result in a list */
	public static List<String> readToList(Reader r) throws IOException {
		try ( BufferedReader in = new BufferedReader(r) ) {
			List<String> l = new ArrayList<>();
			String line = null;
			while ((line = in.readLine()) != null)
				l.add(line);
			return Collections.unmodifiableList(l);
		}
	}




	/** @return true if i is an even number */
	public static boolean isEven(int i) { return (i&1)==0; }
	/** @return true if i is an odd number */
	public static boolean isOdd(int i) { return !isEven(i); }

	/** Read the lines (as UTF8) of the resource file fn from the package of the given class into a (unmodifiable) list of strings
	 * @throws IOException */
	public static List<String> readResource(Class<?> clazz, String fn) throws IOException {
		InputStream in =  clazz.getResourceAsStream(fn);
		if (in == null) {
			throw new FileNotFoundException("InputStream is null for " + fn);
		}
        if (Path.of(fn).endsWith(".gz")) {
        	in = new GZIPInputStream(in);
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        List<String> lines = new ArrayList<String>();
        String line;
        while ((line = reader.readLine()) != null) {
        	lines.add(line);
        }
        return lines;
	}

	
	/** Get a file to read the raw contents of the given resource :) */
  public static File getResourceAsFile(Class<?> clazz, String fn) throws IOException {
    URL url = clazz.getResource(fn);
    if (url == null || url.getFile() == null) {
      throw new IOException("resource \"" + fn + "\" relative to " + clazz + " not found.");
    }
    return new File(url.getFile());
  }


	/** Read the lines (as UTF8) of the resource file fn from the package of the given class into a string */
	public static String readResourceToStringChecked(Class<?> clazz, String fn) throws IOException {
		InputStream in =  clazz.getResourceAsStream(fn);
		if (in == null) {
			throw new FileNotFoundException("InputStream is null for " + fn);
		}
        if (Path.of(fn).endsWith(".gz")) {
        	in = new GZIPInputStream(in);
        }
		return new String(in.readAllBytes(), StandardCharsets.UTF_8);
	}
}
