package uk.ac.soton.ecs.twtk;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
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
            String testResult = testing.classify(testImage).getPredictedClasses().iterator().next();
            writer.write(id);
            writer.write(" ");
            writer.write(testResult);
            writer.newLine();
        }
        writer.close();
    }
}
