package uk.ac.soton.ecs.twtk;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;


public class App
{
    public static void main(String[] args) throws IOException
    {
    	boolean testRun = false;
    	boolean profile = false;
    	String runNum = "";
    	if(args.length >= 2)
    	{
    		if(args[0].equalsIgnoreCase("debug"))
    		{
    			testRun = true;
    		}
    		else if(args[0].equalsIgnoreCase("profile"))
    		{
    			profile = true;
    		}
    		runNum = args[1];
    	}
    	else if(args.length == 1)
    	{
    		runNum = args[0];
    	}
    	
    	TestHarness harness;
        int runNo = Integer.parseInt(runNum);
    	switch(runNo)
    	{
    		case 1:
    			if(profile) 
    			{
    				profileK();
    				return;
    			}
    			harness = new TestHarness(new KNNClassifier(19), "run1.txt");
    			break;
    		case 2:
    			harness = new TestHarness(new LinearClassifier(), "run2.txt");
    			break;
    		case 3:
    			harness = new TestHarness(new BestClassifier(), "run3.txt");
    			break;
    		default:
    			harness = null;
    			break;
    	}
    	if(harness == null)
        {
            System.out.println("Invalid run number!");
            return;
        }
        //TestHarness run1Harness = new TestHarness(new KNNClassifier(), "run1.txt");
        //run1Harness.testRun(20, 20);
        //profileK();
        if(testRun) harness.testRun(5, 5, true);
        else harness.run();
    }

    private static void profileK() throws IOException
    {
        int N = 25;
        int noIterations = 5;

        FileOutputStream outputStream = new FileOutputStream("test-results.csv");
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream));

        writer.write("run,k-value,no_iterations,averaged ratio");
        writer.newLine();

        for (int i=1; i<=9; i++)
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
