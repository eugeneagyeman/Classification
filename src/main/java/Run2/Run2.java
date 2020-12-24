package Run2;

import org.openimaj.data.DataSource;
import org.openimaj.data.DoubleArrayBackedDataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/** To do list:
 *  > create 8x8 patches, sample every 4 pixels in x,y direction
 *  > Take pixels from patches and potentially mean centre and normalise (working with intensities, not gradient-magnitude)
 *  > flatten each patch into a vector
 *  > map each patch to a representative visual word via K means algorithm
 *  > build classifier that takes into account number of visual words in the image and classify accordingly
 * Key info:
 * > Vocab / codebook size , try 500 as the size of the codebook
 * > Use LiblinearAnnotator
 *
 */
public class Run2 {

    final static int SAMPLE_SIZE = 100; //100 for testing, recommended 8
    final static int SAMPLE_GAP = 30; //30 for testing, recommended 4
    final static int MAX_FEATURE_CAP = 10; //probs like 5000-10000 or something

    public static void main(String[] args) throws IOException {
        Run2 testRun = new Run2();
        File workingDir = new File("../classification");

        VFSGroupDataset<FImage> dataset = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/training.zip", ImageUtilities.FIMAGE_READER);
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter(dataset,
                5, //50 normally
                0,
                5); //50 normally

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testData = splits.getTestDataset();

        //testing 1d array thingy
        List<double[]> testing1dArray = testRun.patchMaker(trainingData.get("bedroom").get(0));
        if (testing1dArray.get(0).equals(testing1dArray.get(1))){
            System.out.println("the same");
        }else{
            System.out.println("not the same");
        }


        //testing picture sampling to see if it works
        /*
        DisplayUtilities.displayName(train.get("bedroom").get(0),"testImage");
        ArrayList<double[][]> testPatches = testRun.patchMakerTest(train.get("bedroom").get(0));

        for(double[][] eachImg: testPatches){
            float[][] img = new float[SAMPLE_SIZE][SAMPLE_SIZE];
            for (int px = 0; px < SAMPLE_SIZE; px++){
                for (int py = 0; py < SAMPLE_SIZE; py++){
                    img[px][py] = ((float) eachImg[px][py]);
                }
            }
            DisplayUtilities.display(new FImage(img));
        }
         */


    }



    //returns an arraylist of pixel patches for the image parsed in
    public List<double[][]> patchMakerTest(FImage img) {
        //size of image
        int imgRows = img.getRows();
        int imgCols = img.getCols();

        //double[][] patchTemplate = new double[SAMPLE_SIZE][SAMPLE_SIZE];
        ArrayList<double[][]> allPatches = new ArrayList<>();
        double[][][] ttt = new double[1000][SAMPLE_SIZE][SAMPLE_SIZE];

        //int linearCounter = 0;

        for (int pixelPointerX = 0; pixelPointerX < imgRows; pixelPointerX += SAMPLE_GAP) {
            if ((pixelPointerX + SAMPLE_SIZE - 1) < imgRows) { //checking if there is space vertically for template

                for (int pixelPointerY = 0; pixelPointerY < imgCols; pixelPointerY += SAMPLE_GAP) {
                    if ((pixelPointerY + SAMPLE_SIZE - 1) < imgCols) { //checking if there is space horizontally
                        double[][] patchTemplate = new double[SAMPLE_SIZE][SAMPLE_SIZE];

                        for (int patchPointerX = 0; patchPointerX < SAMPLE_SIZE; patchPointerX++) {
                            for (int patchPointerY = 0; patchPointerY < SAMPLE_SIZE; patchPointerY++) {
                                patchTemplate[patchPointerX][patchPointerY] = img.pixels[pixelPointerX + patchPointerX][pixelPointerY + patchPointerY];
                            }
                        }

                        allPatches.add(patchTemplate);

                    }


                }
            }

        }

        System.out.println(allPatches.size());

        if(allPatches.size() > MAX_FEATURE_CAP){
            allPatches = new ArrayList<>(allPatches.subList(0,MAX_FEATURE_CAP));
        }
        System.out.println(allPatches.size());

        return allPatches;

    }


    //returns an arraylist of pixel patches for the image parsed in
    public List<double[]> patchMaker(FImage img) {
        //size of image
        int imgRows = img.getRows();
        int imgCols = img.getCols();

        double[][] patchTemplate = new double[SAMPLE_SIZE][SAMPLE_SIZE];
        ArrayList<double[]> allPatches = new ArrayList<>();

        for (int pixelPointerX = 0; pixelPointerX < imgRows; pixelPointerX += SAMPLE_GAP) {
            if ((pixelPointerX + SAMPLE_SIZE - 1) < imgRows) { //checking if there is space vertically for template

                for (int pixelPointerY = 0; pixelPointerY < imgCols; pixelPointerY += SAMPLE_GAP) {
                    if ((pixelPointerY + SAMPLE_SIZE - 1) < imgCols) { //checking if there is space horizontally

                        int linearCounter = 0;
                        double[] patch = new double[SAMPLE_SIZE * SAMPLE_SIZE];

                        for (int patchPointerX = 0; patchPointerX < patchTemplate.length; patchPointerX++) {
                            for (int patchPointerY = 0; patchPointerY < patchTemplate[patchPointerX].length; patchPointerY++) {
                                patch[linearCounter] = img.pixels[pixelPointerX + patchPointerX][pixelPointerY + patchPointerY];
                                linearCounter++;
                            }
                        }
                        allPatches.add(patch);
                    }


                }
            }
        }

        //caps features out to what ever max feature cap is
        //basically first x features will be used -> rest of image is not included
        //System.out.println(allPatches.size());
        if(allPatches.size() > MAX_FEATURE_CAP){
            allPatches = new ArrayList<>(allPatches.subList(0,MAX_FEATURE_CAP));
        }

        //System.out.println(allPatches.size());

        /*
        for (int i = 0; i <  MAX_FEATURE_CAP; i++) {
            System.out.println(allPatches.get(i));
        }
         */

        return allPatches;

    }

    //look into concatenate
    //crop and flatten

    //get patch
    //flatten
    //then add to a big array of patch

    //1 -> 2d array then flatten to 1d array -> using feature vectors
    //2 -> single pointer to 1d array instead of using patchpointerx calculation

    //look into zeropadding the image to get all possible samples from image
    //

    //borrowed from em's code
    public static double[] meanCentredVector(double[] vector) {
        if (vector.length == 0)
            return vector;

        int s = 0;
        for (double v : vector) {
            s+=v;
        }
        double m = s/vector.length;
        double[] newV = new double[vector.length];
        for (int i = 0; i < newV.length; i++) {
            newV[i] = vector[i] - m;
        }

        return newV;
    }

    public static double[] normaliseVector(double[] vector) {
        DoubleFV fv = normaliseVector(new DoubleFV(vector));
        return fv.values;
    }

    public static DoubleFV normaliseVector(DoubleFV fv) {
        return fv.normaliseFV();
    }

    //Use if you want to create padded images when sampling
    //Implementation to account for padding not been made.
    public FImage paddedImage(FImage imgToBePadded){
        return imgToBePadded.padding(SAMPLE_SIZE,SAMPLE_SIZE,0f);
    }


}
