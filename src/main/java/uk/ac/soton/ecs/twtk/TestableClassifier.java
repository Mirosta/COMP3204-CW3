package uk.ac.soton.ecs.twtk;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;

/**
 * Created by Tom on 02/12/2014.
 */
public interface TestableClassifier
{
    public void setup();

    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingSet);

    public ClassificationResult<String> classify(FImage image);
}
