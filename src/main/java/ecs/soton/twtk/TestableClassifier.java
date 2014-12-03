package ecs.soton.twtk;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.image.FImage;

/**
 * Created by Tom on 02/12/2014.
 */
public interface TestableClassifier
{
    public void setup();

    public void train(Dataset<FImage> trainingSet);

    public String classify(FImage image);
}
