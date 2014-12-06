package uk.ac.soton.ecs.twtk;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

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
    
    
	// Extracts first 10000 dense SIFT features from the images in the dataset
	// and then clusters them into 300 separate classes.
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
			GroupedDataset<String, ? extends ListDataset<FImage>, FImage> sample, LinearExtractor featureExtractor)
	{
		List<FImage> images = new ArrayList<FImage>();
	    
	    for (FImage i : sample) {
	    	featureExtractor.extractFeature(i);
	    }
	
//	    return result.defaultHardAssigner();
	}
	
	static class LinearExtractor implements FeatureExtractor<DoubleFV, FImage> {
		
		ListDataset<FImage> images = new ListDataset<FImage>();
		
		@Override
		public DoubleFV extractFeature(FImage object) {
			
			int imgHeight = object.getHeight();
			int imgWidth = object.getWidth();
			int N = 8;		// N x N dimensions
			
			// Adjust height and width to be equal, then get center of image
			for (int y=0; y<imgHeight; y += 4) {
				for (int x=0; x<imgWidth; x += 4) {
					// mean-center here
					images.add(object.extractROI(x, y, N, N).normalise());
				}
			}
			
			
			// Get pixel vector
			double[] pixelVector = result.getDoublePixelVector();
			
			
			DoubleFV featureVector = new DoubleFV(pixelVector);
			return featureVector.normaliseFV();
		}
	}
}
