package uk.ac.soton.ecs.twtk;

import de.bwaldvogel.liblinear.SolverType;

import org.openimaj.data.DataSource;
import org.openimaj.data.DoubleArrayBackedDataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntDoublePair;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Tom on 08/12/2014.
 */
public class BestClassifierGIST implements TestableClassifier
{
    private LiblinearAnnotator<FImage, String> annotator;
    private Gist<FImage> gist;

    @Override
    public void setup()
    {
    	int[] b = {16, 16, 16, 16};
        gist = new Gist<FImage>(256, 256, Gist.DEFAULT_ORIENTATIONS_PER_SCALE, true);
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingSet)
    {
        HardAssigner<double[], double[], IntDoublePair> assigner = trainQuantiser(trainingSet, gist);
        HomogeneousKernelMap hKMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Uniform);
        FeatureExtractor<DoubleFV, FImage> extractor = hKMap.createWrappedExtractor(new GistExtractor(gist, assigner));
        annotator = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        annotator.train((GroupedDataset<String, ListDataset<FImage>, FImage>) trainingSet);
    }

    private HardAssigner<double[], double[], IntDoublePair> trainQuantiser(
            Dataset<FImage> sample, Gist<FImage> gist)
    { 
        List<DoubleFV> featureVectors = new ArrayList<DoubleFV>();
        
        int n = 1;
        int length = sample.numInstances();

        //Foreach image in the training dataset, analyse the image and add the keypoints to the list
        for (FImage d : sample)
        {	
            System.out.println("Training " + n + "/" + length);
            gist.analyseImage(d);
            featureVectors.add(gist.getResponse().asDoubleFV());
            n++;
        }


        DoubleKMeans km = DoubleKMeans.createKDTreeEnsemble(50); 
        System.out.println("Created kMeans Tree");
        
        DataSource<double[]> datasource = new DoubleArrayBackedDataSource(makeDoubleArray(featureVectors));
        DoubleCentroidsResult result = km.cluster(datasource);
        System.out.println("Clustered");
        return result.defaultHardAssigner();
    }
    
    private double[][] makeDoubleArray(List<DoubleFV> doubles)
    {
        double[][] output = new double[doubles.size()][];

        int n = 0;
        for (DoubleFV subArr : doubles)
        {
            output[n] = subArr.asDoubleVector();
            n++;
        }

        return output;
    }
    
    static class GistExtractor implements FeatureExtractor<DoubleFV, FImage>
    {
        Gist<FImage> gist;
        HardAssigner<double[], ?, ?> assigner;

        public GistExtractor(Gist<FImage> gist, HardAssigner<double[], ?, ?> assigner)
        {
            this.gist = gist;
            this.assigner = assigner;
        }

        //Extract the features from a given image using the assigner trained earlier
        public DoubleFV extractFeature(FImage image)
        {
        	image = image.normalise();
        	gist.analyseImage(image);
            DoubleFV featureVector = gist.getResponse().asDoubleFV();
            return featureVector;
        }
    }

    @Override
    public ClassificationResult<String> classify(FImage image)
    {
        return annotator.classify(image);
    }
}
