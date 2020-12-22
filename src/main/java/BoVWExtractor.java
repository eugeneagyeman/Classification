import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
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

        getFVArrs(imgs, dps);

        MAX_DP_SIZ = 10000;
        if (dps.size() > MAX_DP_SIZ)
            datapoints = dps.subList(0, MAX_DP_SIZ);
        else
            datapoints = dps;

        float[][] fvs = datapoints.toArray(new float[datapoints.size()][64]);
//        DataSource<float[]> ds = new LocalFeatureListDataSource<>(dps);
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        FloatKMeans.Result r = km.cluster(fvs);

        return r.defaultHardAssigner();
    }

    // yeah probably
    // you could just checkout my branch
    // one sec

    private static void getFVArrs(GroupedDataset<String, ListDataset<FImage>, FImage> imgs, List<float[]> dps) {
        imgs.forEach( (c, fs) -> {
            fs.forEach( f -> {
//            FImage f = fs.getRandomInstance();
                float[][] pxls = f.pixels;
                for (int x = 0; x < pxls.length; x+=4) {
                    for (int y = 0; y < pxls.length; y+=4) {
                        float[] fv = getPatch(f, x, y);
                        dps.add(getPatch(f, x, y));
                    }
                }
            });
        });
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

        VFSGroupDataset<FImage> dataset = new VFSGroupDataset<>("zip:" + workingDir.getAbsolutePath() + "/training.zip", ImageUtilities.FIMAGE_READER);
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter(dataset,
                50,
                0,
                50);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testData = splits.getTestDataset();


        BoVWExtractor bovwe = new BoVWExtractor();
        bovwe.ass = trainHardAssigner(trainingData);

        LiblinearAnnotator<FImage, String> lla = new LiblinearAnnotator<>(bovwe,
                LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L2R_LR,
                20,
                25);

        lla.train(trainingData);

//        trainingData.forEach( (c, fs) -> {
//            System.out.println(c+" "+fs.size());
//        });

        // main issue is that I think I'm making an "Office" classifier, need more
        // probably one for each class idk... probs not an annotator for each?
        // Well the list *[(Office, 0.0625)]* should have more than one elem, with confidence for each
        // currently just office at 6% confidence...

//        FImage i = trainingData.get("Office").get(0);
//        DisplayUtilities.display(i);
//        System.out.println(lla.annotate(i));

        trainingData.forEach( (c, fs) -> {
//            fs.forEach( f -> {
            FImage f = fs.getRandomInstance();
//            DisplayUtilities.display(f);
            System.out.println(c + " -" + lla.annotate(f));
//            });
        });

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
//        BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 2,2);
//        return spatial.aggregate(things, img.getBounds()).normaliseFV();
        return bovw.aggregate(things).normaliseFV();
    }
}
