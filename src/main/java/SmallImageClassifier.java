import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.clustering.assignment.soft.DoubleKNNAssigner;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * OK so next challenge is to use the KNNClassifier classes - feed data and read from indices
 * Work out the ouput - NOT ACCURATE (maybe use another classifier to work it out?)
 */
public class SmallImageClassifier {

    final static int BOX_SIZE_FACTOR = 8;
    final static int CROP_SIZE_PX = 16;

    public static DoubleFV getSmallImageFV(FImage img) {
        FImage croppedImage;
        DoubleFV featureVector;

        final int cropW = img.width/ BOX_SIZE_FACTOR;
        final int cropH = img.height/ BOX_SIZE_FACTOR;
        final int centreX=img.width/2;
        final int centreY=img.height/2;

        final int boxSiz = (cropW < cropH)  ? cropW : cropH;

        croppedImage=img.extractROI(centreX-(cropW/2), centreY-(cropH/2), boxSiz, boxSiz);
        croppedImage.processInplace(new ResizeProcessor(CROP_SIZE_PX,CROP_SIZE_PX));
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

    private static String findMajority(int[] neighbours, double[][] dataset,  Map<double[], String> classes) {
        Map<String, Integer> occur = new HashMap<>();
        for (int neighbourI : neighbours) {
            double[] neighbour = dataset[neighbourI];
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

    public static void main(String[] args) throws IOException {
        File workingDir = new File("../classification");
        VFSGroupDataset<FImage> groups = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/training.zip", ImageUtilities.FIMAGE_READER);
        int splitSize = 75; // ideal as each group has 100 images --> 50 each
        System.out.println("splitSize = " + splitSize);

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(groups, splitSize, 0, splitSize);
        GroupedDataset<String, ListDataset<FImage>, FImage> train = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();

        Map<double[], String> classesPairs = new HashMap<>();

        List<double[]> featureSpace = new ArrayList<>();

        train.forEach((s, fs) -> {
            fs.forEach(f -> {
                double[] fv = getSmallImageFVArr(f);
                featureSpace.add(fv);
                classesPairs.put(fv, s);
            });
        });

        System.out.println("featureSpace.size() = " + featureSpace.size());
        final int K = (int) Math.sqrt(featureSpace.size());
        double[][] ds = featureSpace.toArray(new double[featureSpace.size()][CROP_SIZE_PX*CROP_SIZE_PX]);

        DoubleKNNAssigner nn = new DoubleKNNAssigner(ds, DoubleFVComparison.EUCLIDEAN, K);

        final double[] correct = {0};
        final double[] incorrect = {0};

        test.forEach((s, fs) -> {
            fs.forEach((f) -> {
                double[] fv = getSmallImageFVArr(f);
                int[] r = nn.assign(fv);
                String classifier = findMajority(r, ds, classesPairs);
                if (classifier.equals(s))
                    correct[0]++;
                else
                    incorrect[0]++;
                classesPairs.put(fv, classifier);
            });
        });

        System.out.println("REPORT:\n" +
                "Correct: " + correct[0] + "\n" +
                "Incorrect: " + incorrect[0] + "\n" +
                "Accuracy: " + (100 * correct[0] / (correct[0] + incorrect[0])) + "%");
        System.out.println("CROP_FACTOR = " + BOX_SIZE_FACTOR);
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
