package uk.ac.soton.ecs.twtk;

import java.io.IOException;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App
{
    public static void main( String[] args ) throws IOException
    {
        TestHarness run2Harness = new TestHarness(new LinearClassifier(), "run2.txt");
        run2Harness.run();
    }
}
