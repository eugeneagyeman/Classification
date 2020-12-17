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


    public static class NNDataPoint {
        String classifier;
        DoubleFV fv;

        public NNDataPoint(String classifier, DoubleFV featureVector) {
            this.classifier = classifier;
            this.fv = featureVector;
        }

        public double compareTo(DoubleFV featureVector) {
            return this.fv.compare(featureVector, DoubleFVComparison.EUCLIDEAN);
        }

        public String getClassifier() {
            return classifier;
        }

        public String toString() {
            return classifier+":"+fv.length();
        }
    }

    public static DoubleFV getSmallImageFV(FImage img) {
        FImage croppedImage;
        DoubleFV featureVector;

        final int cropW = img.width/4;
        final int cropH = img.height/4;
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
//        FImage img = ImageUtilities.readF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
//        FImage croppedImage;
//
//        final int cropW = img.width/4;
//        final int cropH = img.height/4;
//        final int centreX=img.width/2;
//        final int centreY=img.height/2;
//
//        final int boxSiz = (cropW < cropH)  ? cropW : cropH;
//
//        croppedImage=img.extractROI(centreX-(cropW/2), centreY-(cropH/2), boxSiz, boxSiz);
//        DisplayUtilities.display(img);
//        DisplayUtilities.display(croppedImage);
        File workingDir = new File("../classification");
        VFSGroupDataset<FImage> groups = new VFSGroupDataset<>("zip:"+workingDir.getAbsolutePath()+"/training.zip", ImageUtilities.FIMAGE_READER);
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(groups, groups.size()/2, 0, groups.size()/2);
        GroupedDataset<String, ListDataset<FImage>, FImage> train = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();

        Map<FImage, double[]> featureSpace = new HashMap<>();
        Map<String, Integer> classIntPairs = new HashMap<>();

        double[][] ds = new double[train.size()][256];
        final int[] i = {0};
        final int K = 16;

        train.forEach( (s, fs) -> {
            FImage image = fs.getRandomInstance();
            featureSpace.put(image, getSmallImageFVArr(image));
            ds[i[0]++] = getSmallImageFVArr(image);
        });


        DoubleKNNAssigner nn = new DoubleKNNAssigner(ds, DoubleFVComparison.EUCLIDEAN, K);
        GroupedDataset<String, ListDataset<FImage>, FImage> testing = splitter.getTestDataset();

        DoubleNearestNeighboursExact dnn = new DoubleNearestNeighboursExact(ds);

        test.forEach( (s, fs) -> {
            System.out.println("---"+s+"---");
            FImage f = fs.get(0);
            int[] r = nn.assign(getSmallImageFVArr(f));
            printListWithIndexes(r);

        });

//        train.forEach( (s, fs) -> {
//            System.out.println("-------"+s+"-------");
//            fs.forEach( f -> {
////                printListWithIndexes(nn.assign(getSmallImageFVArr(f)));
//                int[][] index = new int[1][K];
//                double[][] distances = new double[1][K];
//                dnn.searchKNN(new double[][] {getSmallImageFVArr(f)}, K, index, distances);
//                System.out.println("is");
//                for (int[] ints : index) {
//                    System.out.print("\t");printListWithIndexes(ints);
//                }
//                System.out.println("ds");
//                for (double[] ints : distances) {
//                    System.out.print("\t");printListWithIndexes(ints);
//                }
//            });
//            System.out.println("--------------");
//        });

        /*

        int[] assignment = assignRandomImage(nn, testing);
        printListWithIndexes(assignment);

        assignment = assignRandomImage(nn, testing);
        printListWithIndexes(assignment);*/

        /*// train based on stuff I guess
        List<NNDataPoint> featureSpace = new ArrayList<>();
        train.forEach( (s, fs) -> {
//            System.out.println("####");
            fs.forEach( f -> {
//                System.out.println("s = " + s);
                featureSpace.add(new NNDataPoint(s, getSmallImageFV(f)));
            });
//            System.out.println("####");
        });

        System.out.println(featureSpace);

        final int K = 5;
        for (int i = 1; i < 50; i++) {
            System.out.println("-------------\ni = " + i);
            dothething(test, featureSpace, i);
        }*/


//        List<double[]> ds = new ArrayList<>();
//        List<int[]> assigns = new ArrayList<>();
//        List<double[]> weights = new ArrayList<>();
//
//        final int[] i = {0};
//        train.forEach( (s, fs) -> {
//            System.out.println("s = " + s);
//
//            fs.forEach( f -> {
//                ds.add(getSmallImageFVArr(f));
//                double[] w = {1};
//                int[] a = {i[0]};
////                Arrays.fill(a, i[0]);
////                Arrays.fill(w, 1);
//                weights.add(w);
//                assigns.add(a);
//            });
//
//            classIntPairs.put(s, i[0]++);
//        });
//
//        double[][] datapoints = ds.toArray(new double[0][]);
//        int[][] dataassignments = assigns.toArray(new int[0][]);
//        double[][] dataweights = weights.toArray(new double[0][]);
//
//        nn.assignWeighted(datapoints, dataassignments, dataweights);

//        System.out.println("classIntPairs = " + classIntPairs);




//        featureSpace.forEach( (i, fv) -> {
//            final FImage[] closest = {null, i};
//            final Double[] minDist = {Double.MAX_VALUE};
//            featureSpace.forEach( (i2, fv2) -> {
//                if (!i.equals(i2)) {
//                    Double dist = fv.compare(fv2, DoubleFVComparison.EUCLIDEAN);
//                    if (dist < minDist[0] || closest[0] == null) {
//                        minDist[0] = dist;
//                        closest[0] = i2;
//                    }
//                }
//            });
//            DisplayUtilities.display(minDist[0].toString(), closest);
//        });


    }

    private static void dothething(GroupedDataset<String, ListDataset<FImage>, FImage> test, List<NNDataPoint> featureSpace, int K) {
        final int[] incorrect = {0};
        final int[] correct = {0};

        // test based on test set with known classifiers
        test.forEach( (s, fs) -> {
            fs.forEach( f -> {
//                DisplayUtilities.display(f);
//                System.out.println("--------------");
                Map<Double, String> distances = new TreeMap<>();
                NNDataPoint nnp = new NNDataPoint(s, getSmallImageFV(f));
                featureSpace.forEach( dp -> {
//                    System.out.println(dp.compareTo(nnp.fv));
                    distances.put(dp.compareTo(nnp.fv), dp.classifier);
                });
                String[] vals = distances.values().toArray(new String[K]);
                Map<String, Integer> classOccurences = new HashMap<>(vals.length);
//                System.out.println(distances.entrySet());
                for (int k = 0; k < K; k++) {
                    if (vals[k] == null)
                        continue;

                    classOccurences.computeIfPresent(vals[k], (str, i) -> i+1);
                    classOccurences.putIfAbsent(vals[k], 1);
                }
                final String[] max = new String[1];
                final int[] maxOcc = {0};
                classOccurences.forEach( (c, o) -> {
                    if (o > maxOcc[0]) {
                        maxOcc[0] = o;
                        max[0] = c;
                    }
                });
//                System.out.println(classOccurences);
//                System.out.println(featureSpace);
//                System.out.println(max[0]+" occurred "+maxOcc[0]+" times");
                if (max[0].equals(s))
                    correct[0]++;
                else
                    incorrect[0]++;
//                DisplayUtilities.display(max[0], f);
            });
        });

        System.out.println(correct[0]+" correct");
        System.out.println(incorrect[0]+" incorrect");
        System.out.println(correct[0]/incorrect[0]+" ratio");
    }

    private static int[] assignRandomImage(DoubleKNNAssigner nn, GroupedDataset<String, ListDataset<FImage>, FImage> testing) {
        FImage test = testing.getRandomInstance();
        return nn.assign(getSmallImageFVArr(test));
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

    private static void printZeroIndex(int[] assignment) {
        for (int n = 0; n < assignment.length; n++) {
            int ass = assignment[n];
            if (ass == 0) {
                System.out.println("i:"+n+" = "+ass);
                break;
            }
        }
    }

}
