package uk.ac.soton.ecs.twtk;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.poi.hssf.record.formula.functions.T;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.*;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.*;
import org.openimaj.image.feature.local.aggregate.FisherVector;
import org.openimaj.math.statistics.distribution.MixtureOfGaussians;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.gmm.GaussianMixtureModelEM;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by Tom on 08/12/2014.
 */
public class BestClassifier implements TestableClassifier
{
    PyramidDenseSIFT<FImage> pdsift;
    MixtureOfGaussians siftMixtureModel;
    LiblinearAnnotator<FImage, String> annotator;

    @Override
    public void setup()
    {
        DenseSIFT dsift = new DenseSIFT(5, 7); //3, 7 is slow
        pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7); //4, 6, 8, 10 is slow
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingSet)
    {
        List<LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>> siftPoints = siftAnalysis(trainingSet, pdsift);
        siftMixtureModel = trainMixtureModel(siftPoints);
        System.out.println("Trained gmm");
        HomogeneousKernelMap hKMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, FImage> extractor = hKMap.createWrappedExtractor(new PHOWExtractor(pdsift, siftMixtureModel));
        annotator = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        annotator.train((GroupedDataset<String, ListDataset<FImage>, FImage>) trainingSet);
    }

    private List<LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>> siftAnalysis(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift)
    {
        List<LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>> allKeys = new ArrayList<LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>>();

        int n = 1;
        int length = sample.numInstances();

        //Foreach image in the training dataset, analyse the image and add the keypoints to the list
        for (FImage rec : sample)
        {
            System.out.println("Training " + n + "/" + length);
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allKeys.add(toDouble(pdsift.getFloatKeypoints(0.005f)));

            n++;
        }

        //Take the first 10,000 keypoints
        if (allKeys.size() > 10000)
            allKeys = allKeys.subList(0, 10000);
        return allKeys;
    }

    /*private LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>> toDouble (LocalFeatureList<FloatDSIFTKeypoint> keypoints, boolean useless)
    {
        LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>> newList = new MemoryLocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>();
        for(LocalFeature<SpatialLocation, FloatFV> feature : keypoints)
        {
            newList.add(new LocalFeatureImpl<SpatialLocation, DoubleFV>(feature.getLocation(), feature.getFeatureVector().asDoubleFV()));
        }
        return newList;
    }*/

    private static LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>> toDouble (LocalFeatureList<? extends LocalFeature<SpatialLocation, FloatFV>> keypoints)
    {
        LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>> newList = new MemoryLocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>();
        for(LocalFeature<SpatialLocation, FloatFV> feature : keypoints)
        {
            newList.add(new LocalFeatureImpl<SpatialLocation, DoubleFV>(feature.getLocation(), feature.getFeatureVector().asDoubleFV()));
        }
        return newList;
}

    private MixtureOfGaussians trainMixtureModel(List<LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>> allKeys)
    {
        //Stores the keypoints from analysing the images
        GaussianMixtureModelEM gmm = new GaussianMixtureModelEM(5, GaussianMixtureModelEM.CovarianceType.Diagonal);
        //Cluster the keypoints into a bag of visual words, and return an assigner which can assign new keypoints into that bag
        DataSource<double[]> datasource = new LocalFeatureListDataSource<LocalFeature<SpatialLocation, DoubleFV>, double[]>(allKeys);;
        double[][] dataBuffer = new double[datasource.numRows()][];
        datasource.getData(0, datasource.numRows(), dataBuffer);
        datasource = null; //Clear up duplicated memory
        System.out.println("Training gmm");
        return gmm.estimate(dataBuffer);
    }

    static class CombinedExtractor implements FeatureExtractor<DoubleFV, FImage>
    {
        Collection<FeatureExtractor<DoubleFV, FImage>> extractors;

        public CombinedExtractor(Collection<FeatureExtractor<DoubleFV, FImage>> extractors)
        {
            this.extractors = extractors;
        }

        @Override
        public DoubleFV extractFeature(FImage fImage)
        {
            DoubleFV outVector = null;
            for(FeatureExtractor<DoubleFV, FImage> extractor : extractors)
            {
                DoubleFV feature = extractor.extractFeature(fImage);
                outVector = combineFeatures(outVector, feature);
            }

            return outVector;
        }

        private DoubleFV combineFeatures(DoubleFV a, DoubleFV b)
        {
            if(a == null) return b;
            if(b == null) return a;
            double[] arrA = a.asDoubleVector();
            double[] arrB = b.asDoubleVector();

            if(arrB.length > arrA.length)
            {
                double[] tempArr = arrA;
                arrA = arrB;
                arrB = tempArr;
            }

            for(int i = 0; i < arrA.length; i++)
            {
                arrA[i] = arrA[i] + arrB[i];
            }

            return new DoubleFV(arrA);
        }
    }

    //A Pyramid Histogram of Words feature extractor
    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>
    {
        PyramidDenseSIFT<FImage> pdsift;
        MixtureOfGaussians mixtureModel;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, MixtureOfGaussians mixtureModel)
        {
            this.pdsift = pdsift;
            this.mixtureModel = mixtureModel;
        }

        //Extract the features from a given image using the assigner trained earlier
        public DoubleFV extractFeature(FImage image)
        {

            FisherVector<double[]> fv  = new FisherVector<double[]>(mixtureModel, true);
            //BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            //More accurate but at the cost of increased run tim

            return fv.aggregate(toDouble(pdsift.getFloatKeypoints(0.015f))).normaliseFV();
        }
    }

    @Override
    public ClassificationResult<String> classify(FImage image)
    {
        return annotator.classify(image);
    }
}
