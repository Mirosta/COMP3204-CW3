package ecs.soton.twtk;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.image.FImage;

/**
 * Created by Tom on 02/12/2014.
 */
public class LinearClassifier implements TestableClassifier
{
    @Override
    public void setup()
    {

    }

    @Override
    public void train(Dataset<FImage> trainingSet)
    {

    }

    @Override
    public String classify(FImage image)
    {
        return "Test";
    }
}
