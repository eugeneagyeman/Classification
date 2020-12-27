import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.DataUtils;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.svm.SVMAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.time.Timer;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Run3Classifier {

    final static int CLUSTERS = 300;
    final static int STEPS = 5;
    final static int BIN = 7;
    final static float MAGFACTOR = 6f;
    final static int SIZES = 7;
    final static float ENERGYTHRESH1 = 0.005f;
    final static float ENERGYTHRESH2 = 0.015f;
    final static int FEATURES = 10000;
    final static int SAMPLE = 30;


    //TODO: Edit this to remove or possibly implement
    private static final String CACHE_PATH = "src/main/java/cache";
    private static final Timer timer = new Timer();

    static {

        String dir = new File("../classification").getAbsolutePath();
        System.out.println("Running Classifier...");
        try {
            //Record Implementation
            //Unzip training and testing files
            //For each folder in training
                //create new list backdataset
                //convert images into identifiable records

            VFSGroupDataset<FImage> trainImages =
                    new VFSGroupDataset<>("zip:" + dir + "/training.zip", ImageUtilities.FIMAGE_READER);

            VFSListDataset<FImage> testImages = new VFSListDataset<FImage>("zip:" + dir + "/testing.zip", ImageUtilities.FIMAGE_READER);

            GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(trainImages, 15, false);

            GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(data, 80, 0, 20);
            GroupedDataset<String, ListDataset<FImage>, FImage> subTrainingSet = splits.getTrainingDataset();
            GroupedDataset<String, ListDataset<FImage>, FImage> subTestSet = splits.getTestDataset();

            //SIFTs
            DenseSIFT denseSIFT = new DenseSIFT(STEPS, BIN); //5 7
            PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<>(denseSIFT, MAGFACTOR, SIZES);

            HardAssigner<byte[], float[], IntFloatPair> assigner;

            assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(subTrainingSet, SAMPLE), pyramidDenseSIFT); //30
            FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pyramidDenseSIFT, assigner);

            NaiveBayesAnnotator<FImage, String> naiveBayesAnnotator = new NaiveBayesAnnotator<FImage, String>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);

            timer.start();
            naiveBayesAnnotator.train(subTrainingSet);
            timer.stop();
            long resultantTime = timer.duration();
            System.out.println("Time: " + resultantTime);


            subTestSet.forEach((c, fs) -> {
                FImage img = fs.getRandomInstance();
                System.out.println("Actual: " + c + "\nExpected: " + naiveBayesAnnotator.annotate(img));
                System.out.println();
            });

            for (int i = 0; i < 60; i++) {
                int num = i;
                FImage image = testImages.getRandomInstance();
                DisplayUtilities.displayName(image, "" + num);
                System.out.println(num + ": " + naiveBayesAnnotator.annotate(image));
            }

            ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<CMResult<String>, String, FImage>
                    (naiveBayesAnnotator, subTestSet, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

            Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
            CMResult<String> result = eval.analyse(guesses);
            System.out.println(result.getDetailReport());
            System.out.println("Classification Completed.");
            System.out.println("CLUSTERS: " + CLUSTERS);
            System.out.println("STEPS: " + STEPS);
            System.out.println("BIN: " + BIN);
            System.out.println("MAGFACTOR: " + MAGFACTOR);
            System.out.println("SIZES: " + SIZES);
            System.out.println("ENERGYTHRESH1: " + ENERGYTHRESH1);
            System.out.println("ENERGYTHRESH2: " + ENERGYTHRESH2);
            System.out.println("FEATURES: " + FEATURES);
            System.out.println("SAMPLE: " + SAMPLE);
//            System.out.println(eval.getExpected());
//            Map<FImage, Set<String>> expected = eval.getExpected();
//            System.out.println(expected.size());
//            final static int CLUSTERS = 300;
//            final static int STEPS = 4;
//            final static int BIN = 8;
//            final static float MAGFACTOR = 6f;
//            final static int SIZES = 7;
//            final static float ENERGYTHRESH = 0.015f;
//            final static int FEATURES = 10000;
//            final static int SAMPLE = 30;

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> sample, PyramidDenseSIFT<FImage> pyramidDenseSIFT) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new
                ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
        for (FImage img : sample) {
            pyramidDenseSIFT.analyseImage(img);
            allkeys.add(pyramidDenseSIFT.getByteKeypoints(ENERGYTHRESH1));//0.005f
        }
        if (allkeys.size() > FEATURES)//10000
            allkeys = allkeys.subList(0, FEATURES);//10000
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);//300
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair>
                assigner) {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage fImage) {
            pdsift.analyseImage(fImage);
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(
                    bovw, 2, 2);
            return spatial.aggregate(pdsift.getByteKeypoints(ENERGYTHRESH2), fImage.getBounds()).normaliseFV();//0.015f
        }

    }
}