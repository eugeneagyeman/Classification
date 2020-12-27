import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BoVWExtractor implements FeatureExtractor<DoubleFV, FImage> {

    private static int MAX_DP_SIZ;

    public static float[] getPatch(FImage img, int x, int y) {
        float[][] vs = img.pixels;
        float[] patch = new float[64];
        int ppoint = 0;

        for (int i = x; i < x+8; i++) {
            for (int j = y; j < y+8; j++) {
                float v = 0;
                try {
                    v = vs[x+i][y+j];
                } catch (IndexOutOfBoundsException e) {
                    v = 0;
                } finally {
                    patch[ppoint++]= v;
                }
            }
        }
        return patch;
    }

    public static FloatFV getPatchFV(FImage img, int x, int y) {
        float[][] vs = img.pixels;
        float[] patch = new float[64];
        int ppoint = 0;

        for (int i = x; i < x+8; i++) {
            for (int j = y; j < y+8; j++) {
                float v = 0;
                try {
                    v = vs[x+i][y+j];
                } catch (IndexOutOfBoundsException e) {
                    v = 0;
                } finally {
                    patch[ppoint++]= v;
                }
            }
        }
        return new FloatFV(patch);
    }

    public static FloatKeypoint getKeyPoint(float x, float y, float ori, float scale, float[] vec) {
        return new FloatKeypoint(x, y,ori, scale, vec);
    }

    public static HardAssigner<float[], float[], IntFloatPair> trainHardAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> imgs) {
        List<float[]> dps = new ArrayList<>();
        List<float[]> datapoints;

        imgs.forEach(( c, fs) -> {
            fs.forEach( f -> {
                dps.addAll(getFVs(f));
            });
        });

        MAX_DP_SIZ = 10000;
        if (dps.size() > MAX_DP_SIZ)
            datapoints = dps.subList(0, MAX_DP_SIZ);
        else
            datapoints = dps;

        float[][] fvs = datapoints.toArray(new float[datapoints.size()][64]);
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        FloatCentroidsResult r = km.cluster(fvs);

        return r.defaultHardAssigner();
    }

    // yeah probably
    // you could just checkout my branch
    // one sec

    private static List<float[]> getFVs(FImage f) {
        List<float[]> dps = new ArrayList<>();
        float[][] pxls = f.pixels;
        for (int x = 0; x < pxls.length; x+=4) {
            for (int y = 0; y < pxls.length; y+=4) {
                dps.add(getPatch(f, x, y));
            }
        }

        return dps;
    }

    private static void getFVKPS(GroupedDataset<String, ListDataset<FImage>, FImage> imgs, List<FloatKeypoint> dps) {
        imgs.forEach( (c, fs) -> {
            fs.forEach( f -> {
//            FImage f = fs.getRandomInstance();
                List<FloatKeypoint> keypoints = getFloatKeypoints(f);
                dps.addAll(keypoints);
            });
        });
    }

    private static List<FloatKeypoint> getFloatKeypoints(FImage f) {
        float[][] pxls = f.pixels;
        List<FloatKeypoint> keypoints = new ArrayList<>();
        for (int x = 0; x < pxls.length; x+=4) {
            for (int y = 0; y < pxls.length; y+=4) {
                float[] fv = getPatch(f, x, y);
                FloatKeypoint keyPoint = getKeyPoint(x, y, 0, 1, getPatch(f, x, y));
                keypoints.add(keyPoint);
            }
        }
        return keypoints;
    }

    public static void main(String[] args) throws FileSystemException {
        File workingDir = new File("../classification");
        System.out.print("Retrieving training images... ");
        int nTrain = 50;
        VFSGroupDataset<FImage> dataset = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/training.zip", ImageUtilities.FIMAGE_READER);
        dataset.remove("training");
        System.out.println("Retrieved.");
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter(dataset,
                nTrain,
                0,
                nTrain);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testData = splits.getTestDataset();

        long time = System.nanoTime();

        System.out.print("Creating assigner... ");
        BoVWExtractor bovwe = new BoVWExtractor();
        bovwe.ass = trainHardAssigner(trainingData);
        System.out.println("Created.");

        // -vars-with 50-50----|results l2rl2loss|--results mcsvmcs--|--results l2r--|
        // c = 1, eps at 1e-4 -> 22%, 22%, 22%   | 48%, 47%, 46%     |  17%, 16%, ..
        // c = 10, eps at 1e-4 -> 33%, 32%, 35%  |    ...            | ...
        // c = 15, eps at 1e-4 -> 35%, 38%, 36%  |    ...            | ...
        // c = 20, eps at 1e-4 -> 36%, 37% ...   | 44%, 49%, 46%     | ...
        // c = 25, eps at 1e-4 ->      ...       |                   | ...
        System.out.print("Initialising annotator... ");
        LiblinearAnnotator<FImage, String> lla = new LiblinearAnnotator<>(bovwe,
                LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.MCSVM_CS,
                1,
                .00001);
        System.out.println("Initialised.");
        System.out.print("Training annotator... ");
        lla.train(trainingData);
        System.out.println("Trained. Generated annotations:");

        System.out.println(lla.getAnnotations());

        System.out.print("Annotating test set... ");
        final int[] corr = {0};
        final int[] incorr = {0};
        testData.forEach( (c, fs) -> {
            fs.forEach( f -> {
                List<ScoredAnnotation<String>> annotations = lla.annotate(f);
                if (c.equals(annotations.iterator().next().annotation))
                    corr[0]++;
                else
                    incorr[0]++;
            });
        });
        System.out.println("Annotated.");
        double correct = corr[0];
        double incorrect = incorr[0];

        System.out.println("REPORT:\n" +
                "Correct: "+correct+"\n" +
                "Incorrect: "+incorrect+"\n" +
                "Accuracy: "+100*(correct/(correct+incorrect)));

        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<>(
                        lla, splits.getTestDataset(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        // print results to sdtout
        System.out.println(result.getDetailReport());

        time -= System.nanoTime(); // calc diff
        time *= -.000000001; // convert to seconds
        System.out.println("Completed in "+time+" seconds.");

    }

    private static void printList(List<ScoredAnnotation<String>> list) {
        if (list == null)
            return;
        list.forEach( a -> {
            System.out.println("a.annotation = " + a.annotation);
        });
    }

    HardAssigner<float[], float[], IntFloatPair> ass;

    @Override
    public DoubleFV extractFeature(FImage img) {
        BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(ass);

        List<FloatKeypoint> things = getFloatKeypoints(img);
        BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 4,4);
        return spatial.aggregate(things, img.getContentArea()).normaliseFV();
//        return bovw.aggregate(things).normaliseFV();
    }
}
