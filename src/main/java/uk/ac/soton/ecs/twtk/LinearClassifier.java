package uk.ac.soton.ecs.twtk;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.lang.ArrayUtils;
import org.openimaj.data.DataSource;
import org.openimaj.data.FloatArrayBackedDataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.math.geometry.point.Point2d;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

/**
 * Created by Tom on 02/12/2014.
 */
public class LinearClassifier implements TestableClassifier
{
    private PatchExtractor patchExtractor;
    private LiblinearAnnotator<FImage, String> annotator;

    @Override
    public void setup()
    {
        patchExtractor = new PatchExtractor(8, 4);
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> trainingSet)
    {
        System.out.println("Training assigner");
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(trainingSet, patchExtractor);
        System.out.println("Training annotator");
        annotator = new LiblinearAnnotator<FImage, String>(new LinearExtractor(assigner, patchExtractor), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        annotator.train((GroupedDataset<String, ListDataset<FImage>, FImage>) trainingSet);
    }

    @Override
    public ClassificationResult<String> classify(FImage image)
    {
        return annotator.classify(image);
    }


    private HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> sample, PatchExtractor patchExtractor)
    {
        List<float[]> imagePatches = new ArrayList<float[]>();

        int n = 1;
        for (FImage i : sample)
        {
            System.out.println(n++ + "/" + sample.numInstances());
            List<LocalFeature<SpatialLocation, FloatFV>> patches = patchExtractor.getPatches(i);
            for (LocalFeature<SpatialLocation, FloatFV> patch : patches)
                imagePatches.add(patch.getFeatureVector().getVector());
        }

        //Cluster up to 100000 randomly selected features into 500 partitions, and return an assigner which can assign new features into those partitions
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        FloatArrayBackedDataSource datasource = new FloatArrayBackedDataSource(makeFloatArray(imagePatches));
        float[][] randomSamples = new float[Math.min(100000, imagePatches.size())][patchExtractor.patchSize*patchExtractor.patchSize];
        datasource.getRandomRows(randomSamples);
        //Clear space in memory
        imagePatches = null;
        datasource = null;

        System.out.println("Clustering");
        FloatCentroidsResult result = km.cluster(randomSamples);
        System.out.println("Clustered");
        return result.defaultHardAssigner();
    }

    private float[][] makeFloatArray(List<float[]> floats)
    {
        float[][] output = new float[floats.size()][];

        int n = 0;
        for (float[] subArr : floats)
        {
            output[n] = subArr;
            n++;
        }

        return output;
    }

    static class PatchExtractor
    {
        private int patchSize;
        private int patchSpacing;

        public PatchExtractor(int patchSize, int patchSpacing)
        {
            this.patchSize = patchSize;
            this.patchSpacing = patchSpacing;
        }

        List<FImage> getPatchImages(FImage image)
        {
            int imgHeight = image.getHeight();
            int imgWidth = image.getWidth();
            List<FImage> patches = new ArrayList<FImage>();

            for (int y = 0; y < imgHeight; y += patchSpacing)
            {
                for (int x = 0; x < imgWidth; x += patchSpacing)
                {
                    FImage patch = image.extractROI(x, y, patchSize, patchSize);
                    //Mean center and normalise
                    float average = patch.sum() / (patchSize * patchSize);
                    patch = patch.subtract(average).normalise();
                    patches.add(patch);
                }
            }

            return patches;
        }

        List<LocalFeature<SpatialLocation, FloatFV>> getPatches(FImage image)
        {
            int imgHeight = image.getHeight();
            int imgWidth = image.getWidth();
            List<LocalFeature<SpatialLocation, FloatFV>> patches = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

            for (int y = 0; y < imgHeight; y += patchSpacing)
            {
                for (int x = 0; x < imgWidth; x += patchSpacing)
                {
                    FImage patch = image.extractROI(x, y, patchSize, patchSize);
                    //Mean center and normalise
                    float average = patch.sum() / (patchSize * patchSize);
                    patch = patch.subtract(average).normalise();
                    patches.add(new LocalFeatureImpl<SpatialLocation, FloatFV>(new SpatialLocation(x, y), new FloatFV(flattenImage(patch))));
                }
            }

            return patches;
        }

        float[] flattenImage(FImage image)
        {
            float[] output = new float[image.getWidth() * image.getHeight()];

            for (int y = 0; y < image.getHeight(); y++)
            {
                for (int x = 0; x < image.getWidth(); x++)
                {
                    output[y * image.getWidth() + x] = image.getPixel(x, y);
                }
            }

            return output;
        }
    }

    static class LinearExtractor implements FeatureExtractor<SparseIntFV, FImage>
    {
        private HardAssigner<float[], float[], IntFloatPair> assigner;
        private PatchExtractor patchExtractor;

        public LinearExtractor(HardAssigner<float[], float[], IntFloatPair> assigner, PatchExtractor patchExtractor)
        {
            this.assigner = assigner;
            this.patchExtractor = patchExtractor;
        }

        @Override
        public SparseIntFV extractFeature(FImage image)
        {
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
            List<LocalFeature<SpatialLocation, FloatFV>> features = patchExtractor.getPatches(image);

            return bovw.aggregate(features);
        }
    }
}
