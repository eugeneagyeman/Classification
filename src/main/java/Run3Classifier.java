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
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.BatchAnnotator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LinearSVMAnnotator;
import org.openimaj.ml.annotation.svm.SVMAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.time.Timer;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

public class Run3Classifier {

    final static int CLUSTERS = 300; //300
    final static int STEPS = 4; //5
    final static int BIN = 8; //7
    final static float MAGFACTOR = 6f; //6f
    final static int SIZES = 7; //7
    final static float ENERGYTHRESH1 = 0.005f; //0.005f
    final static float ENERGYTHRESH2 = 0.015f;  //0.015f
    final static int FEATURES = 10000; //10000
    final static int SAMPLE = 30; //30
    final static int TRAININGSIZE = 50;
    final static int TESTINGSIZE = 50;
    final static int VALIDATIONSIZE = 0;
    final static int BLOCKX = 2;//2
    final static int BLOCKY = 2;//2
    final static HashMap<String, String> annotations = new HashMap<>();
    final static HashMap<String, String> NBannotations = new HashMap<>();

    //TODO: Edit this to remove or possibly implement
    private static final String CACHE_PATH = "src/main/java/cache";
    private static final Timer timer = new Timer();

    static {
        final Writer fileWriter = new Writer(3);
        String dir = new File("").getAbsolutePath();
        try {
            System.out.println("Running Classifier...");
            VFSGroupDataset<FImage> trainImages =
                    new VFSGroupDataset<>("zip:" + dir + "/training.zip", ImageUtilities.FIMAGE_READER);

            VFSListDataset<FImage> testImages = new VFSListDataset<>("zip:" + dir + "/testing.zip", ImageUtilities.FIMAGE_READER);

            GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(trainImages, 15, false);
            GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(data, TRAININGSIZE, VALIDATIONSIZE, TESTINGSIZE);
            GroupedDataset<String, ListDataset<FImage>, FImage> subTrainingSet = splits.getTrainingDataset();
            GroupedDataset<String, ListDataset<FImage>, FImage> subTestSet = splits.getTestDataset();

            //SIFTs
            DenseSIFT denseSIFT = new DenseSIFT(STEPS, BIN); //5 7
            PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<>(denseSIFT, MAGFACTOR, SIZES);

            HardAssigner<byte[], float[], IntFloatPair> assigner; //30
            assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(subTrainingSet, SAMPLE), pyramidDenseSIFT);

            HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
            FeatureExtractor<DoubleFV, FImage> wrappedExtractor = hkm.createWrappedExtractor(new PHOWExtractor(pyramidDenseSIFT, assigner));

            NaiveBayesAnnotator<FImage, String> naiveBayesAnnotator = new NaiveBayesAnnotator<>(wrappedExtractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
            LiblinearAnnotator<FImage, String> liblinearAnnotator = new LiblinearAnnotator<>(wrappedExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

            //Training Classifier using Naives Bayes
            timer.start();
            naiveBayesAnnotator.train(subTrainingSet);
            timer.stop();
            long resultantTime = timer.duration();
            System.out.println("Time for Naive Bayes Training: " + convertToMinutes(resultantTime) + " minutes");

            //Training Classifier using Lib Linear
            timer.start();
            liblinearAnnotator.train(subTrainingSet);
            timer.stop();
            long resultantTime3 = timer.duration();
            System.out.println("Time for LibLinear Training: " + convertToMinutes(resultantTime3) + " minutes");


            //Annotating Images using Naive Bayes Classifier
            timer.start();
            annotateImagesNB(testImages, naiveBayesAnnotator);
            timer.stop();
            long resultantTime4 = timer.duration();
            getNaivesBayesEvaluationResult(subTestSet, naiveBayesAnnotator, resultantTime4);


            //Annotating Images using the LibLinear Classifier
            timer.start();
            annotateImagesLL(testImages, liblinearAnnotator);
            timer.stop();
            long resultantTime6 = timer.duration();
            getLibLinearEvaluationResult(subTestSet, liblinearAnnotator, resultantTime6);
            fileWriter.writeResults(annotations);

            printParameters();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void printParameters() {
        System.out.println("--------------------------");
        System.out.println("--------------------------");
        System.out.println("Classification complete.");
        System.out.println("CLUSTERS: " + CLUSTERS);
        System.out.println("STEPS: " + STEPS);
        System.out.println("BIN: " + BIN);
        System.out.println("MAGFACTOR: " + MAGFACTOR);
        System.out.println("SIZES: " + SIZES);
        System.out.println("ENERGYTHRESH1: " + ENERGYTHRESH1);
        System.out.println("ENERGYTHRESH2: " + ENERGYTHRESH2);
        System.out.println("FEATURES: " + FEATURES);
        System.out.println("SAMPLE: " + SAMPLE);
        System.out.println("TRAININGSIZE: " + TRAININGSIZE);
        System.out.println("TESTINGSIZE: " + TESTINGSIZE);
        System.out.println("VALIDATIONSIZE: " + VALIDATIONSIZE);
        System.out.println("BLOCKX: " + BLOCKX);
        System.out.println("BLOCKY: " + BLOCKY);
    }

    private static CMResult<String> getLibLinearEvaluationResult(GroupedDataset<String, ListDataset<FImage>, FImage> subTestSet, LiblinearAnnotator<FImage, String> liblinearAnnotator, long duration) {
        ClassificationEvaluator<CMResult<String>, String, FImage> eval3 = new ClassificationEvaluator<>
                (liblinearAnnotator, subTestSet, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> guesses3 = eval3.evaluate();
        CMResult<String> result = eval3.analyse(guesses3);

        System.out.println("Lib Linear Annotator:");
        System.out.println(result.getDetailReport());
        System.out.println("Time for Naive Bayes Report: " + convertToMinutes(duration) + " minutes");
        System.out.println("--------------------------");
        return result;
    }

    private static CMResult<String> getNaivesBayesEvaluationResult(GroupedDataset<String, ListDataset<FImage>, FImage> subTestSet, NaiveBayesAnnotator<FImage, String> naiveBayesAnnotator, long duration) {
        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>
                (naiveBayesAnnotator, subTestSet, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println("Naive Bayes Annotator:");
        System.out.println(result.getDetailReport());
        System.out.println("Time for Naive Bayes Report: " + convertToMinutes(duration) + " minutes");
        System.out.println("--------------------------");
        return result;
    }

    private static void annotateImagesNB(VFSListDataset<FImage> testImages, Annotator<FImage, String> naiveBayesAnnotator) {
        for (int i = 0; i < testImages.numInstances(); i++) {
            String imageName = getImageNameFromFile(testImages.getID(i));
            FImage img = testImages.getInstance(i);
            String annotation = naiveBayesAnnotator.annotate(img).get(0).annotation;
            NBannotations.put(imageName, annotation);
        }
    }

    private static void annotateImagesLL(VFSListDataset<FImage> testImages, Annotator<FImage, String> liblinearAnnotator) {
        IntStream.range(0, testImages.numInstances()).forEach(i -> {
            String imageName = getImageNameFromFile(testImages.getID(i));
            FImage img = testImages.getInstance(i);
            String annotation = liblinearAnnotator.annotate(img).get(0).annotation;
            annotations.put(imageName, annotation);
        });
    }

    private static long convertToMinutes(long duration) {
        return duration / 60000;
    }

    private static String getImageNameFromFile(String name) {
        return name.substring(8);
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> sample, PyramidDenseSIFT<FImage> pyramidDenseSIFT) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new
                ArrayList<>();
        for (FImage img : sample) {
            pyramidDenseSIFT.analyseImage(img);
            allkeys.add(pyramidDenseSIFT.getByteKeypoints());//0.005f
        }
        if (allkeys.size() > FEATURES)//10000
            allkeys = allkeys.subList(0, FEATURES);//10000
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);//300
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
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
                    bovw, BLOCKX, BLOCKY); //2 2
            return spatial.aggregate(pdsift.getByteKeypoints(ENERGYTHRESH2), fImage.getBounds()).normaliseFV();//0.015f
        }
    }
}