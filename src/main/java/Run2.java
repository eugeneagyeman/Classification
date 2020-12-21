import org.apache.commons.vfs2.FileObject;
import org.apache.hadoop.ha.HealthCheckFailedException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.clustering.assignment.soft.DoubleKNNAssigner;

import java.awt.image.SampleModel;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    final static int SAMPLE_SIZE = 100;
    final static int SAMPLE_GAP = 30;

    public static void main(String[] args) throws IOException {
        Run2 testRun = new Run2();
        File workingDir = new File("../classification");
        VFSGroupDataset<FImage> train = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/training.zip", ImageUtilities.FIMAGE_READER);
        int splitSize = 75; // ideal as each group has 100 images --> 50 each
        System.out.println("splitSize = " + splitSize);

        VFSListDataset<FImage> test = new VFSListDataset<>("zip:" + workingDir.getAbsolutePath() + "/testing.zip", ImageUtilities.FIMAGE_READER);
        VFSGroupDataset<FImage> tests = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/testing.zip", ImageUtilities.FIMAGE_READER);

        //testing picture
        //DisplayUtilities.displayName(train.get("bedroom").get(0),"testImage");
        //ArrayList<float[][]> testPatches = testRun.patchMakerTest(train.get("bedroom").get(0));

        //testing 1d array thingy
        ArrayList<float[]> testing1dArray = testRun.patchMaker(train.get("bedroom").get(0));
        if (testing1dArray.get(0).equals(testing1dArray.get(1))){
            System.out.println("the same");
        }else{
            System.out.println("not the same");
        }

        /*
        for(int i = 0 ; i < testPatches.size() ; i++){
            FImage imgTest = new FImage(testPatches.get(i));
            DisplayUtilities.displayName(imgTest, "Patch"+i);
            //DisplayUtilities.display(imgTest);
            System.out.println("Patch " + i + " : " +testPatches.get(i));
        }

         */

    }


    /*
    //returns an arraylist of pixel patches for the image parsed in
    public ArrayList<float[][]> patchMakerTest(FImage img) {
        //size of image
        int imgRows = img.getRows();
        int imgCols = img.getCols();

        //float[][] patchTemplate = new float[SAMPLE_SIZE][SAMPLE_SIZE];
        ArrayList<float[][]> allPatches = new ArrayList<>();
        float[][][] ttt = new float[1000][SAMPLE_SIZE][SAMPLE_SIZE];

        //int linearCounter = 0;

        for (int pixelPointerX = 0; pixelPointerX < imgRows; pixelPointerX += SAMPLE_GAP) {
            if ((pixelPointerX + SAMPLE_SIZE - 1) < imgRows) { //checking if there is space vertically for template

                for (int pixelPointerY = 0; pixelPointerY < imgCols; pixelPointerY += SAMPLE_GAP) {
                    if ((pixelPointerY + SAMPLE_SIZE - 1) < imgCols) { //checking if there is space horizontally
                        float[][] patchTemplate = new float[SAMPLE_SIZE][SAMPLE_SIZE];

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

        return allPatches;

    }
    */

    //returns an arraylist of pixel patches for the image parsed in
    public ArrayList<float[]> patchMaker(FImage img) {
        //size of image
        int imgRows = img.getRows();
        int imgCols = img.getCols();

        float[][] patchTemplate = new float[SAMPLE_SIZE][SAMPLE_SIZE];
        ArrayList<float[]> allPatches = new ArrayList<>();

        for (int pixelPointerX = 0; pixelPointerX < imgRows; pixelPointerX += SAMPLE_GAP) {
            if ((pixelPointerX + SAMPLE_SIZE - 1) < imgRows) { //checking if there is space vertically for template

                for (int pixelPointerY = 0; pixelPointerY < imgCols; pixelPointerY += SAMPLE_GAP) {
                    if ((pixelPointerY + SAMPLE_SIZE - 1) < imgCols) { //checking if there is space horizontally

                        int linearCounter = 0;
                        float[] patch = new float[SAMPLE_SIZE * SAMPLE_SIZE];

                        for (int patchPointerX = 0; patchPointerX < patchTemplate.length; patchPointerX++) {
                            for (int patchPointerY = 0; patchPointerY < patchTemplate[patchPointerX].length; patchPointerY++) {
                                patch[linearCounter] = img.pixels[pixelPointerX + patchPointerX][pixelPointerY + patchPointerY]; //
                                linearCounter++;
                            }
                        }
                        allPatches.add(patch);
                    }


                }
            }
        }

        //System.out.println(allPatches.size());
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




}
