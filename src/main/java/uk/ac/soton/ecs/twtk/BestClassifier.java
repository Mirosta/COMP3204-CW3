package uk.ac.soton.ecs.twtk;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Tom on 08/12/2014.
 */
public class BestClassifier implements TestableClassifier
{
    PyramidDenseSIFT<FImage> pdsift;
    HardAssigner<byte[], float[], IntFloatPair> assigner;
    LiblinearAnnotator<FImage, String> annotator;

    @Override
    public void setup()
    {
        DenseSIFT dsift = new DenseSIFT(3, 7);
        pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 4, 6, 8, 10);
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingSet)
    {
        assigner = trainQuantiser(trainingSet, pdsift);
        HomogeneousKernelMap hKMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, FImage> extractor = hKMap.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
        annotator = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        annotator.train((GroupedDataset<String, ListDataset<FImage>, FImage>) trainingSet);
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift)
    {
        //Stores the keypoints from analysing the images
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        int n = 1;
        int length = sample.numInstances();

        //Foreach image in the training dataset, analyse the image and add the keypoints to the list
        for (FImage rec : sample)
        {
            System.out.println("Training " + n + "/" + length);
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));

            n++;
        }

        //Take the first 10,000 keypoints
        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        //Create a KMeans clusterer with 600 visual words, is more accurate than 300 but takes longer to create the assigner
        //In my tests it moved from 0.4 accuracy to 0.477
        //But the assigner moved from just over 4 minutes to create to over 7 minutes
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(600); //Originally
        System.out.println("Created kMeans Tree");
        //Cluster the keypoints into a bag of visual words, and return an assigner which can assign new keypoints into that bag
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);
        System.out.println("Clustered");
        return result.defaultHardAssigner();
    }

    //A Pyramid Histogram of Words feature extractor
    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>
    {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        //Extract the features from a given image using the assigner trained earlier
        public DoubleFV extractFeature(FImage image)
        {
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            //More accurate but at the cost of increased run time
            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2); //Originally 2, 4

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }

    @Override
    public ClassificationResult<String> classify(FImage image)
    {
        return annotator.classify(image);
    }
}
