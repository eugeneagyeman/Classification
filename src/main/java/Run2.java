import org.apache.commons.vfs2.FileObject;
import org.apache.hadoop.ha.HealthCheckFailedException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.clustering.assignment.soft.DoubleKNNAssigner;

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
    final static int SAMPLEGAP = 30;

    public static void main(String[] args) throws IOException {
        Run2 testRun = new Run2();
        File workingDir = new File("../classification");
        VFSGroupDataset<FImage> train = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/training.zip", ImageUtilities.FIMAGE_READER);
        int splitSize = 75; // ideal as each group has 100 images --> 50 each
        System.out.println("splitSize = " + splitSize);

        VFSListDataset<FImage> test = new VFSListDataset<>("zip:" + workingDir.getAbsolutePath() + "/testing.zip", ImageUtilities.FIMAGE_READER);
        VFSGroupDataset<FImage> tests = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/testing.zip", ImageUtilities.FIMAGE_READER);
        System.out.println("Hi");

        /*
        for (int i = 0; i<20; i++){
            DisplayUtilities.display(train.get("bedroom").get(i));
        }*/
        DisplayUtilities.displayName(train.get("bedroom").get(0),"testImage");
        ArrayList<float[][]> testPatches;
        testPatches = testRun.patchMakerTest(train.get("bedroom").get(0));


    }

    //returns an arraylist of pixel patches for the image parsed in
    public ArrayList<float[][]> patchMakerTest(FImage img) {
        //size of image
        int imgRows = img.getRows();
        int imgCols = img.getCols();

        float[][] patchTemplate = new float[SAMPLE_SIZE][SAMPLE_SIZE];
        ArrayList<float[][]> allPatches = new ArrayList<>();

        for (int pixelPointerX = 0; pixelPointerX < imgRows; pixelPointerX += SAMPLEGAP) {
            if ((pixelPointerX + patchTemplate.length - 1) < imgRows) { //checking if there is space vertically for template

                for (int pixelPointerY = 0; pixelPointerY < imgCols; pixelPointerY += SAMPLEGAP) {
                    if ((pixelPointerY + patchTemplate[0].length - 1) < imgCols) { //checking if there is space horizontally

                        for (int patchPointerX = 0; patchPointerX < patchTemplate.length; patchPointerX++) {
                            for (int patchPointerY = 0; patchPointerY < patchTemplate[0].length; patchPointerY++) {
                                patchTemplate[patchPointerX][patchPointerY] = img.pixels[pixelPointerX + patchPointerX][pixelPointerY + patchPointerY];
                            }
                        }
                        allPatches.add(patchTemplate);
                    }


                }
            }

        }

        for(int i = 0 ; i < 10 ; i++){
            FImage imgTest = new FImage(allPatches.get(i));
            DisplayUtilities.displayName(imgTest, "Patch"+i);
            //DisplayUtilities.display(imgTest);
            System.out.println("Patch " + i + " : " +allPatches.get(i));
        }

        System.out.println(allPatches.size());
        return allPatches;
    }


    //returns an arraylist of pixel patches for the image parsed in
    public ArrayList<float[]> patchMaker(FImage img) {
        //size of image
        int imgRows = img.getRows();
        int imgCols = img.getCols();

        float[][] patchTemplate = new float[SAMPLE_SIZE][SAMPLE_SIZE];
        float[] patch = new float[SAMPLE_SIZE * SAMPLE_SIZE];
        ArrayList<float[]> allPatches = new ArrayList<>();

        for (int pixelPointerX = 0; pixelPointerX < imgRows; pixelPointerX += SAMPLEGAP) {
            if ((pixelPointerX + patchTemplate.length - 1) < imgRows) { //checking if there is space vertically for template

                for (int pixelPointerY = 0; pixelPointerY < imgCols; pixelPointerY += SAMPLEGAP) {
                    if ((pixelPointerY + patchTemplate[0].length - 1) < imgCols) { //checking if there is space horizontally

                        for (int patchPointerX = 0; patchPointerX < patchTemplate.length; patchPointerX++) {
                            for (int patchPointerY = 0; patchPointerY < patchTemplate[patchPointerX].length; patchPointerY++) {
                                patch[patchPointerX * 8 + patchPointerY] = img.pixels[pixelPointerX + patchPointerX][pixelPointerY + patchPointerY]; //
                            }
                        }
                    }

                    allPatches.add(patch);
                }
            }
        }

        System.out.println(allPatches.size());
        return allPatches;
        //look into concatanate
        //crop and flatten

        //get patch
        //flatten
        //then add to a big array of patch

        //1 -> 2d array then flatten to 1d array -> using feature vectors
        //2 -> single pointer to 1d array instead of using patchpointerx calculation

        //look into zeropadding the image to get all possible samples from image
        //

    }




}
