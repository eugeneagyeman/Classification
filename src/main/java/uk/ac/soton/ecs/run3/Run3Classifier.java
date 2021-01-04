package uk.ac.soton.ecs.run3;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.time.Timer;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.main.Writer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class Run3Classifier {

    private final static int CLUSTERS = 300; //300
    private final static int STEPS = 5; //5
    private final static int BIN = 7; //7
    private final static float MAGFACTOR = 6f; //6f
    private final static int SIZES = 8; //7
    private final static float ENERGYTHRESH1 = 0.005f;  //0.005f
    private final static float ENERGYTHRESH2 = 0.015f;  //0.015f
    private final static int FEATURES = 10000; //10000
    private final static int SAMPLE = 30; //30
    private final static int TRAININGSIZE = 80;
    private final static int TESTINGSIZE = 20;
    private final static int VALIDATIONSIZE = 0;
    private final static int BLOCKX = 2;//2
    private final static int BLOCKY = 2;//2
    private final static double HYPERPARMETER_C = 1.0;
    private final static double EPOCHS = 0.00001;
    private final static Timer timer = new Timer();
    private final HashMap<String, String> annotations = new HashMap<>();
    private double accuracy;

    public Run3Classifier() {
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

            //SIFTS
            DenseSIFT denseSIFT = new DenseSIFT(STEPS, BIN); //5 7
            PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<>(denseSIFT, MAGFACTOR, 2, 4, 6, 8, 10);

            HardAssigner<byte[], float[], IntFloatPair> assigner; //30
            assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(subTrainingSet, SAMPLE), denseSIFT);
            HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
            FeatureExtractor<DoubleFV, FImage> wrappedExtractor = hkm.createWrappedExtractor(new PHOWExtractor(pyramidDenseSIFT, assigner));

            LiblinearAnnotator<FImage, String> liblinearAnnotator = new LiblinearAnnotator<>(wrappedExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, HYPERPARMETER_C, EPOCHS);

            //Training Classifier using LibLinear
            timer.start();
            liblinearAnnotator.train(subTrainingSet);
            timer.stop();
            long resultantTime = timer.duration();
            System.out.println("Time for LibLinear Training: " + convertToMinutes(resultantTime) + " minutes");

            //Annotating Images using the LibLinear Classifier
            timer.start();
            annotateImages(testImages, liblinearAnnotator, annotations);
            timer.stop();
            long resultantTime2 = timer.duration();
            getEvaluation(subTestSet, liblinearAnnotator, resultantTime2);

            Writer fileWriter = new Writer("3_" + accuracy);
            fileWriter.writeResults(annotations);

            printParameters();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static long convertToMinutes(long duration) {
        return duration / 60000;
    }

    private static String getImageNameFromFile(String name) {
        return name.substring(8);
    }

    private void annotateImages(VFSListDataset<FImage> testImages, Annotator<FImage, String> liblinearAnnotator, HashMap<String, String> annotations) {
        IntStream.range(0, testImages.numInstances()).forEach(i -> {
            String imageName = getImageNameFromFile(testImages.getID(i));
            FImage img = testImages.getInstance(i);
            String annotation = liblinearAnnotator.annotate(img).get(0).annotation;
            annotations.put(imageName, annotation);
        });

    }

    HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> sample, DenseSIFT denseSIFT) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new
                ArrayList<>();
        for (FImage img : sample) {
            denseSIFT.analyseImage(img);
            allkeys.add(denseSIFT.getByteKeypoints());//0.005f
        }
        if (allkeys.size() > FEATURES)//10000
            allkeys = allkeys.subList(0, FEATURES);//10000
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);//300
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }

    private void printParameters() {
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

    private CMResult<String> getEvaluation(GroupedDataset<String, ListDataset<FImage>, FImage> subTestSet, Annotator<FImage, String> annotator, long annotationDuration) throws IOException {
        System.out.println("Time for Annotation: " + convertToMinutes(annotationDuration) + " minutes");
        ClassificationEvaluator<CMResult<String>, String, FImage> eval3 = new ClassificationEvaluator<>
                (annotator, subTestSet, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

        timer.start();
        Map<FImage, ClassificationResult<String>> guesses3 = eval3.evaluate();
        CMResult<String> result = eval3.analyse(guesses3);
        accuracy = result.getMatrix().getAccuracy();
        timer.stop();
        long duration = timer.duration();

        System.out.println("Annotator Report:");
        System.out.println(result.getDetailReport());
        System.out.println("Time for Report: " + convertToMinutes(duration) + " minutes");
        System.out.println("--------------------------");
        return result;
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
