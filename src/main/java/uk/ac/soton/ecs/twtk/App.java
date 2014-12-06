package uk.ac.soton.ecs.twtk;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;


public class App
{
    public static void main( String[] args ) throws IOException
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
        	TestHarness runHarness = new TestHarness(new KNNClassifier(i), "run" + i + ".txt", true);
        	double totalRatio = 0;
        	for (int n=0; n<noIterations; n++) 
        	{
        		totalRatio += runHarness.testRun(N, N);
        	}
        	System.out.println("Averaged ratio (" + noIterations + " runs): " + totalRatio/noIterations + "\n\n");
        	
        	String csvString = (i-9) + "," + i + "," + noIterations + "," + totalRatio/noIterations;
        	writer.write(csvString);
        	writer.newLine();
        }
        
        writer.close();
        
    }
}
