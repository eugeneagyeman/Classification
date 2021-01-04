package uk.ac.soton.ecs.run1;
import org.apache.commons.vfs2.FileObject;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.clustering.assignment.soft.DoubleKNNAssigner;
import uk.ac.soton.ecs.main.Writer;

import java.io.File;
import java.util.*;

/**
 * Uses KNN to classify images using a feature vector formed by cropping a small square out of the centre of the image.
 * The square is resized to fixed 16x16 pixels and will start at one eighth of the image size.
 *
 * Written by Emily James, Dec 2020
 */
public class SmallImageClassifier {

    final static int BOX_SIZE_FACTOR = 8;
    final static int CROP_SIZE_PX = 16;

    public static DoubleFV getSmallImageFV(FImage img) {
        FImage croppedImage;
        DoubleFV featureVector;

        croppedImage = cropImage(img);

        FloatFV ffv = null;
        for (float[] pixel : croppedImage.pixels) {
            if (ffv == null)
                ffv = new FloatFV(pixel);
            else
                ffv.concatenate(new FloatFV(pixel));
        }
        featureVector = new DoubleFV(ffv.asDoubleVector());
        return featureVector.normaliseFV();
    }

    private static FImage cropImage(FImage img) {
        FImage croppedImage;
        final int cropW = img.width/ BOX_SIZE_FACTOR;
        final int cropH = img.height/ BOX_SIZE_FACTOR;
        final int centreX=img.width/2;
        final int centreY=img.height/2;

        final int boxSiz = (cropW < cropH)  ? cropW : cropH;

        croppedImage=img.extractROI(centreX-(cropW/2), centreY-(cropH/2), boxSiz, boxSiz);
        croppedImage.processInplace(new ResizeProcessor(CROP_SIZE_PX,CROP_SIZE_PX));
        return croppedImage;
    }

    public static double[] getSmallImageFVArr(FImage img) {
        return getSmallImageFV(img).values;
    }

    public static double[] centredNormalisedFV(FImage img) {
        return meanCentredVector(normaliseVector(getSmallImageFVArr(img)));
    }

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

    /**
     * Runs the program from scratch - first training the data using a file called "training.zip"
     * and then generating a list of classes from images in "testing.zip", finally generating its
     * results in "run1.txt". All the files are in the base working directory.
     * @throws FileSystemException
     */
    public void trainAndTest() throws FileSystemException {
        String workingDir = new File("../classification").getAbsolutePath();
        Map<double[], String> classesPairs = new HashMap<>();


        List<double[]> featureSpace = getTrainingVectors(classesPairs, workingDir);

        double[][] fixedFeatureSpace = featureSpace.toArray(new double[featureSpace.size()][CROP_SIZE_PX*CROP_SIZE_PX]);

        DoubleKNNAssigner nn = new DoubleKNNAssigner(fixedFeatureSpace, false, 1);

        Map<String, String> results = classifyTestData(nn, fixedFeatureSpace, classesPairs, workingDir);
        writeResults(results);
    }

    private List<double[]> getTrainingVectors(Map<double[], String> classesPairs, String workingDir) throws FileSystemException {
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>("zip:" + workingDir + "/training.zip", ImageUtilities.FIMAGE_READER);
        List<double[]> featureSpace = new ArrayList<>();

        trainingData.forEach((s, fs) -> {
            System.out.println(s);
            fs.forEach(f -> {
                double[] fv = centredNormalisedFV(f);
                featureSpace.add(fv);
                classesPairs.put(fv, s);
            });
        });
        return featureSpace;
    }

    private Map<String, String> classifyTestData(DoubleKNNAssigner assigner, double[][] fixedFeatureSpace, Map<double[], String> classesPairs, String workingDir) throws FileSystemException {
        VFSGroupDataset<FImage> testData = new VFSGroupDataset<>("zip:" + workingDir + "/testing.zip", ImageUtilities.FIMAGE_READER);
        Map<String, String> results = new TreeMap<>();
        testData.forEach((s,fs) -> {
            for (int i = 0; i < fs.numInstances(); i++) {
                FImage f = fs.getInstance(i);
                FileObject fo = fs.getFileObject(i);

                double[] fv = centredNormalisedFV(f);
                int[] r = assigner.assign(fv);
                String classifier = findMajority(r, fixedFeatureSpace, classesPairs);
                results.put(fo.getName().getBaseName(), classifier);
//                w.writeResult(fo, classifier);
//                w.flush();
            }
        });
        return results;
    }

    private void writeResults(Map<String, String> results) {
        Writer w = new Writer(1);
        w.writeResults(results);
        w.closeFile();
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
