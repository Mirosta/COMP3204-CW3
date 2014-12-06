package uk.ac.soton.ecs.twtk;

import javax.ws.rs.HEAD;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;


public class App
{
    public static void main(String[] args) throws IOException
    {
        //TestHarness run1Harness = new TestHarness(new KNNClassifier(), "run1.txt");
        //run1Harness.testRun(20, 20);
        //profileK();
        TestHarness run2Harness = new TestHarness(new LinearClassifier(), "run2.txt");
        run2Harness.testRun(4, 4, true);
    }

    private static void profileK() throws IOException
    {
        int N = 25;
        int noIterations = 5;

        FileOutputStream outputStream = new FileOutputStream("test-results.csv");
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream));

        writer.write("run,k-value,no_iterations,averaged ratio");
        writer.newLine();

        for (int i=10; i<20; i++)
        {
            System.out.println("Run: " + (i-9) + ", K-value: " + i);
            TestHarness runHarness = new TestHarness(new KNNClassifier(i), "run" + i + ".txt");
            double totalRatio = 0;
            for (int n=0; n<noIterations; n++)
            {
                totalRatio += runHarness.testRun(N, N, false);
            }
            System.out.println("Averaged ratio (" + noIterations + " runs): " + totalRatio/noIterations + "\n\n");

            String csvString = (i-9) + "," + i + "," + noIterations + "," + totalRatio/noIterations;
            writer.write(csvString);
            writer.newLine();
        }

        writer.close();
    }
}
