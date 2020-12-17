import net.didion.jwnl.data.Exc;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.ml.clustering.assignment.soft.DoubleKNNAssigner;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * OK so next challenge is to use the KNNClassifier classes - feed data and read from indices
 * Work out the ouput - NOT ACCURATE (maybe use another classifier to work it out?)
 */
public class SmallImageClassifier {

    static int CROP_FACTOR = 4;

    public static DoubleFV getSmallImageFV(FImage img) {
        FImage croppedImage;
        DoubleFV featureVector;

        final int cropW = img.width/CROP_FACTOR;
        final int cropH = img.height/CROP_FACTOR;
        final int centreX=img.width/2;
        final int centreY=img.height/2;

        final int boxSiz = (cropW < cropH)  ? cropW : cropH;

        croppedImage=img.extractROI(centreX-(cropW/2), centreY-(cropH/2), boxSiz, boxSiz);
        croppedImage.processInplace(new ResizeProcessor(16,16));
        FloatFV ffv = null;
        for (float[] pixel : croppedImage.pixels) {
            if (ffv == null)
                ffv = new FloatFV(pixel);
            else
                ffv.concatenate(new FloatFV(pixel));
        }
        featureVector = new DoubleFV(ffv.asDoubleVector());
        return featureVector;
    }

    public static double[] getSmallImageFVArr(FImage img) {
        return getSmallImageFV(img).values;
    }

    public static void main(String[] args) throws IOException {
            File workingDir = new File("../classification");
            VFSGroupDataset<FImage> groups = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/training.zip", ImageUtilities.FIMAGE_READER);
            int splitSize = 75; // ideal as each group has 100 images --> 50 each
            System.out.println("splitSize = " + splitSize);

            GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(groups, splitSize, 0, splitSize);
            GroupedDataset<String, ListDataset<FImage>, FImage> train = splitter.getTrainingDataset();
            GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();

            List<String> classes = new ArrayList<>();

//        int maxSiz = getMaxSize(train);

            List<double[]> featureSpace = new ArrayList<>();
//        final int K = 5;

            train.forEach((s, fs) -> {
                FImage image = fs.getRandomInstance();
                fs.forEach(f -> {
                    double[] fv = getSmallImageFVArr(image);
                    featureSpace.add(fv);
                    classes.add(s);
                });
            });

            System.out.println("featureSpace.size() = " + featureSpace.size());
            final int K = (int) Math.sqrt(featureSpace.size());
            double[][] ds = featureSpace.toArray(new double[featureSpace.size()][256]);

            DoubleKNNAssigner nn = new DoubleKNNAssigner(ds, DoubleFVComparison.EUCLIDEAN, K);

            final double[] correct = {0};
            final double[] incorrect = {0};

            System.out.println(classes);
            test.forEach((s, fs) -> {
//                System.out.println("---" + s + "---");
                fs.forEach((f) -> {
                    int[] r = nn.assign(getSmallImageFVArr(f));
                    String classifier = findMajority(r, classes);
//                    System.out.println("---==" + classifier + "==---");
                    if (classifier.equals(s))
                        correct[0]++;
                    else
                        incorrect[0]++;
                    classes.add(classifier);
//                    printListWithIndexes(r);
                });
            });

            System.out.println("REPORT:\n" +
                    "Correct: " + correct[0] + "\n" +
                    "Incorrect: " + incorrect[0] + "\n" +
                    "Accuracy: " + (100 * correct[0] / (correct[0] + incorrect[0])) + "%");
            System.out.println("CROP_FACTOR = " + CROP_FACTOR);
    }

    private static String findMajority(int[] neighbours, List<String> classes) {
        Map<String, Integer> occur = new HashMap<>();
        for (int neighbour : neighbours) {
            occur.computeIfPresent(classes.get(neighbour), (s,i) -> (i+1));
            occur.putIfAbsent(classes.get(neighbour),1);
        }

        final String[] maxClass = {null};
        int occurs = 0;
        occur.forEach( (s, c) -> {
            if (c > occurs || maxClass[0] == null) {
                c = occurs;
                maxClass[0] = s;
            }
        });

        return maxClass[0];
    }

    private static void printListWithIndexes(int[] assignment) {
        for (int n = 0; n < assignment.length; n++) {
            int ass = assignment[n];
            System.out.print(ass + /*"[" + n + "]*/"\t");
        }
        System.out.println();
    }

    private static void printListWithIndexes(double[] assignment) {
        for (int n = 0; n < assignment.length; n++) {
            double ass = assignment[n];
            System.out.print(ass + /*"[" + n + "]*/"\t");
        }
        System.out.println();
    }

}
