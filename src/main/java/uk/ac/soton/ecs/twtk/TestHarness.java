package uk.ac.soton.ecs.twtk;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.*;

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
            trainingSet = new VFSGroupDataset<FImage>("zip:" + workingDir + "/training.zip", ImageUtilities.FIMAGE_READER);
            testingSet = new VFSListDataset<FImage>("zip:" + workingDir + "/testing.zip", ImageUtilities.FIMAGE_READER);
        }
        catch (FileSystemException e)
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
        
        for(int i = 0; i < size; i++)
        {
            System.out.println("Testing " + (i+1) + "/" + size);
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

    private static String getClassification(ClassificationResult<? extends String> result)
    {
        return result.getPredictedClasses().iterator().next();
    }

    public void testRun(int noTrainingImages, int noTestingImages)
    {
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(trainingSet, noTrainingImages, 0, noTestingImages);
        testing.setup();
        testing.train(splitter.getTrainingDataset());

        double correctAnswers = 0;
        double incorrectAnswers = 0;
        double totalAnswers = 0;

        for (String actualClassification : splitter.getTestDataset().getGroups())
        {
            System.out.print("Testing group: " + actualClassification);
            ListDataset<FImage> groupInstances = splitter.getTestDataset().getInstances(actualClassification);
            int length = groupInstances.size();
            int n = 1;
            for(FImage image : groupInstances)
            {
                System.out.print(n + "/" + length);
                String classification = getClassification(testing.classify(image));
                if(classification == actualClassification) correctAnswers ++;
                else incorrectAnswers ++;
                totalAnswers ++;
                n++;
            }
        }

        System.out.println("Correct: " + correctAnswers + "/" + totalAnswers);
        System.out.println("Incorrect: " + incorrectAnswers + "/" + totalAnswers);
        System.out.println("Ratio Correct: " + (correctAnswers/totalAnswers));
    }
}
