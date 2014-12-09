package uk.ac.soton.ecs.twtk;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparator;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.processor.ImageProcessor;
import org.openimaj.image.processor.SinglebandImageProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

//Run #1: You should develop a simple k-nearest-neighbour classifier 
//using the "tiny image" feature. The "tiny image" feature is one of the
//simplest possible image representations. One simply crops each image
//to a square about the centre, and then resizes it to a small, fixed
//resolution (we recommend 16x16). The pixel values can be packed into 
//a vector by concatenating each image row. It tends to work slightly 
//better if the tiny image is made to have zero mean and unit length. 
//You can choose the optimal k-value for the classifier.

public class KNNClassifier implements TestableClassifier
{

    private final int k;
    private KNNAnnotator<FImage, String, DoubleFV> knnAnnotator;

    public KNNClassifier(int k)
    {
        this.k = k;
    }

    public void setup()
    {
        TinyImageExtractor extractor = new TinyImageExtractor();
        DoubleFVComparison comparator = DoubleFVComparison.EUCLIDEAN;
        knnAnnotator = KNNAnnotator.create(extractor, comparator, k);
    }


    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingSet)
    {
        knnAnnotator.train(trainingSet);
    }

    @Override
    public ClassificationResult<String> classify(FImage image)
    {
        return knnAnnotator.classify(image);
    }


    static class TinyImageExtractor implements FeatureExtractor<DoubleFV, FImage>
    {

        @Override
        public DoubleFV extractFeature(FImage object)
        {
            FImage img = object.clone();
            int imgHeight = img.getHeight();
            int imgWidth = img.getWidth();

            // Adjust height and width to be equal, then get center of image
            if (imgHeight > imgWidth)
            {
                img = img.extractCenter(imgWidth / 2, imgHeight / 2, imgWidth, imgWidth);
                //DisplayUtilities.display(img);
            }
            else if (imgWidth > imgHeight)
            {
                img = img.extractCenter(imgWidth / 2, imgHeight / 2, imgHeight, imgHeight);
                //DisplayUtilities.display(img);
            }

            int N = 16;        // N x N dimensions

            // Resize image to N x N pixels and normalise
            FImage result = img.process(new ResizeProcessor(N, N));
            float average = result.sum() / (result.getWidth() * result.getHeight());
            result = result.subtract(average);
            result = result.normalise();

            // Get pixel vector
            double[] pixelVector = result.getDoublePixelVector();

            DoubleFV featureVector = new DoubleFV(pixelVector);
            return featureVector;
        }
    }

}
