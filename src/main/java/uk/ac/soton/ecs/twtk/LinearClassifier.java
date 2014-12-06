package uk.ac.soton.ecs.twtk;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingSet)
    {
    }

    @Override
    public ClassificationResult<String> classify(FImage image)
    {
        return new ClassificationResult<String>()
        {
            @Override
            public double getConfidence(String s)
            {
                return 0;
            }

            @Override
            public Set<String> getPredictedClasses()
            {
                List<String> list = new ArrayList<String>();
                list.add("Test");
                return new HashSet<String>(list);
            }
        };
    }
}
