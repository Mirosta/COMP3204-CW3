package uk.ac.soton.ecs.twtk;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Tom on 02/12/2014.
 */
public class TestHarness
{
    private TestableClassifier testing;
    private File testingOutputFile;
    private static VFSGroupDataset<FImage> trainingSet;
    private static VFSListDataset<FImage> testingSet;

    static
    {
        try
        {
            String workingDir = System.getProperty("user.dir");
            trainingSet = new VFSGroupDataset<FImage>("zip:" + workingDir + "/training.zip!/training", ImageUtilities.FIMAGE_READER);
            testingSet = new VFSListDataset<FImage>("zip:" + workingDir + "/testing.zip", ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e)
        {
            System.out.println("Couldn't load training or testing images");
            e.printStackTrace();
        }
    }

    public TestHarness(TestableClassifier testing, String filePath)
    {
        this.testing = testing;
        testingOutputFile = new File(filePath);
    }

    public void run() throws IOException
    {
        testing.setup();
        testing.train(trainingSet);

        FileOutputStream outputStream = new FileOutputStream(testingOutputFile);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream));
        int size = testingSet.size();

        for (int i = 0; i < size; i++)
        {
            System.out.println("Testing " + (i + 1) + "/" + size);
            FImage testImage = testingSet.getInstance(i);
            String[] idParts = testingSet.getID(i).split("/");
            String id = idParts[idParts.length - 1];
            String testResult = getClassification(testing.classify(testImage));
            writer.write(id);
            writer.write(" ");
            writer.write(testResult);
            writer.newLine();
        }
        writer.close();
    }

    private Map<String, ListDataset<FImage>> getMap(VFSGroupDataset<FImage> trainingSet)
    {
        Map<String, ListDataset<FImage>> returnMap = new HashMap<String, ListDataset<FImage>>();

        for(Map.Entry<String, VFSListDataset<FImage>> group : trainingSet.entrySet())
        {
            returnMap.put(group.getKey(), group.getValue());
        }

        return returnMap;
    }

    private static String getClassification(ClassificationResult<? extends String> result)
    {
        if(result.getPredictedClasses().size() == 0)
        {
            System.out.println("No predicted classes!");
            return null;
        }
        return result.getPredictedClasses().iterator().next();
    }

    public double testRun(int noTrainingImages, int noTestingImages, boolean consoleOutput)
    {
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(trainingSet, noTrainingImages, 0, noTestingImages);
        testing.setup();
        testing.train(splitter.getTrainingDataset());

        double correctAnswers = 0;
        double incorrectAnswers = 0;
        double totalAnswers = 0;
        List<Double> correctConfidence = new ArrayList<Double>();
        List<Double> incorrectConfidence = new ArrayList<Double>();

        for (String actualClassification : splitter.getTestDataset().getGroups())
        {
            if (consoleOutput) System.out.println("Testing group: " + actualClassification);
            ListDataset<FImage> groupInstances = splitter.getTestDataset().getInstances(actualClassification);
            int length = groupInstances.size();
            int n = 1;
            for (FImage image : groupInstances)
            {
                if (consoleOutput) System.out.println(n + "/" + length);
                ClassificationResult<String> result = testing.classify(image);
                String classification = getClassification(result);
                if (actualClassification.equals(classification))
                {
                    correctAnswers++;
                    correctConfidence.add(result.getConfidence(classification));
                }
                else
                {
                    incorrectAnswers++;
                    incorrectConfidence.add(result.getConfidence(classification));
                }
                totalAnswers++;
                n++;
            }
        }

        if (consoleOutput) System.out.println("Correct: " + correctAnswers + "/" + totalAnswers);
        if (consoleOutput) System.out.println("Incorrect: " + incorrectAnswers + "/" + totalAnswers);
        if (consoleOutput) System.out.println("Ratio Correct: " + (correctAnswers / totalAnswers));

        return correctAnswers / totalAnswers;
    }
}
