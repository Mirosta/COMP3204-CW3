package ecs.soton.twtk;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.FileNotFoundException;
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
