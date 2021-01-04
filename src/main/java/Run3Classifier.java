import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.experiment.validation.cross.GroupedKFold;
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
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LinearSVMAnnotator;
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

public class Run3Classifier {

    //TODO: Edit this to remove or possibly implement
    private static final String CACHE_PATH = "src/main/java/cache";
    private static final Timer timer = new Timer();
    final static int CLUSTERS = 300; //300
    final static int STEPS = 5; //5
    final static int BIN = 7; //7
    final static float MAGFACTOR = 7f; //6f
    final static int SIZES = 8; //7
    final static float ENERGYTHRESH1 = 0.005f; //0.005f
    final static float ENERGYTHRESH2 = 0.015f;  //0.015f
    final static int FEATURES = 10000; //10000
    final static int SAMPLE = 30; //30
    final static int TRAININGSIZE = 80;
    final static int TESTINGSIZE = 20;
    final static int VALIDATIONSIZE = 0;
    final static int BLOCKX = 2;//2
    final static int BLOCKY = 2;//2
    final static boolean BLOCKSPATIAL = true;


    //TODO:
    static  {


        String dir = new File("").getAbsolutePath();
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

            GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(data, TRAININGSIZE, VALIDATIONSIZE, TESTINGSIZE);
            GroupedDataset<String, ListDataset<FImage>, FImage> subTrainingSet = splits.getTrainingDataset();
            GroupedDataset<String, ListDataset<FImage>, FImage> subTestSet = splits.getTestDataset();

            //SIFTs
            DenseSIFT denseSIFT = new DenseSIFT(STEPS, BIN); //5 7
            PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<>(denseSIFT, MAGFACTOR, 2,4,6,SIZES);

            HardAssigner<byte[], float[], IntFloatPair> assigner;

            timer.start();
            assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(subTrainingSet,SAMPLE), pyramidDenseSIFT); //30
            timer.stop();
            long assignerTime = timer.duration();
            System.out.println("Time for assigner: " + assignerTime / 1000 / 60 + " minutes");

            timer.start();
            HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
            FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pyramidDenseSIFT, assigner);
            FeatureExtractor<DoubleFV, FImage> extractorHkm = hkm.createWrappedExtractor(new PHOWExtractor(pyramidDenseSIFT, assigner));
            timer.stop();
            long extractorTime = timer.duration();
            System.out.println("Time for extractor(s): " + extractorTime / 1000 / 60 + " minutes");

//            NaiveBayesAnnotator<FImage,String> naiveBayesAnnotator = new NaiveBayesAnnotator<FImage, String>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
//            LinearSVMAnnotator<FImage, String> linearSVMAnnotator = new LinearSVMAnnotator<FImage, String>(extractorHkm);
            LiblinearAnnotator<FImage, String> liblinearAnnotator = new LiblinearAnnotator<FImage, String>(extractorHkm, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

            //NB
//            timer.start();
//            naiveBayesAnnotator.train(subTrainingSet);
//            timer.stop();
//            long resultantTime = timer.duration();
//            System.out.println("Time for Naive Bayes: " + resultantTime / 1000 / 60 + " minutes");

//            //SVM
//            timer.start();
//            linearSVMAnnotator.train(subTrainingSet);
//            timer.stop();
//            long resultantTime2 = timer.duration();
//            System.out.println("Time for Linear SVM: " + resultantTime2 / 1000 / 60 + " minutes");

            //LibLinear
            timer.start();
            liblinearAnnotator.train(subTrainingSet);
            timer.stop();
            long resultantTime3 = timer.duration();
            System.out.println("Time for LibLinear: " + resultantTime3 / 1000 / 60 + " minutes");


              //From sub test test
//            subTestSet.forEach((c,fs) -> {
//                FImage img = fs.getRandomInstance();
//                System.out.println("Actual: "+ c +"\nExpected: "+ naiveBayesAnnotator.annotate(img));
//                System.out.println();
//            });


            //random testing data
//            for(int i = 0; i < 60; i++){
//                int num = i;
//                FImage image = testImages.getRandomInstance();
//                DisplayUtilities.displayName(image, ""+num);
//                System.out.println(num + ": " + naiveBayesAnnotator.annotate(image));
//            }

//            //Naive Bayes
//            timer.start();
//            ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<CMResult<String>, String, FImage>
//                    (naiveBayesAnnotator,subTestSet,new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
//            Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
//            CMResult<String> result = eval.analyse(guesses);
//            System.out.println("Naive Bayes Annotator:");
//            System.out.println(result.getDetailReport());
//            timer.stop();
//            long resultantTime4 = timer.duration();
//            System.out.println("Time for Naive Bayes Report: " + resultantTime4 / 1000 / 60 + " minutes");
//            System.out.println("--------------------------");
//
//            //Linear SVM
//            timer.start();
//            ClassificationEvaluator<CMResult<String>, String, FImage> eval2 = new ClassificationEvaluator<CMResult<String>, String, FImage>
//                    (linearSVMAnnotator,subTestSet,new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
//            Map<FImage, ClassificationResult<String>> guesses2 = eval2.evaluate();
//            CMResult<String> result2 = eval2.analyse(guesses2);
//            System.out.println("Linear SVM Annotator:");
//            System.out.println(result2.getDetailReport());
//            timer.stop();
//            long resultantTime5 = timer.duration();
//            System.out.println("Time for Linear SVM Report: " + resultantTime5 / 1000 / 60 + " minutes");
//            System.out.println("--------------------------");

            //LibLinear
            timer.start();
            ClassificationEvaluator<CMResult<String>, String, FImage> eval3 = new ClassificationEvaluator<CMResult<String>, String, FImage>
                    (liblinearAnnotator,subTestSet,new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
            Map<FImage, ClassificationResult<String>> guesses3 = eval3.evaluate();
            CMResult<String> result3 = eval3.analyse(guesses3);
            System.out.println("LibLinear Annotator:");
            System.out.println(result3.getDetailReport());
            timer.stop();
            long resultantTime6 = timer.duration();
            System.out.println("Time for LibLinear Report: " + resultantTime6 / 1000 / 60 + " minutes");
            System.out.println("--------------------------");
            System.out.println("--------------------------");



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
            System.out.println("TRAININGSIZE: " + TRAININGSIZE);
            System.out.println("TESTINGSIZE: " + TESTINGSIZE);
            System.out.println("VALIDATIONSIZE: " + VALIDATIONSIZE);
            System.out.println("BLOCKX: " + BLOCKX);
            System.out.println("BLOCKY: " + BLOCKY);
            System.out.println("BLOCKSPATIAL: " + BLOCKSPATIAL);


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> sample, PyramidDenseSIFT<FImage> pyramidDenseSIFT) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new
                ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
        for (FImage img : sample) {

            pyramidDenseSIFT.analyseImage(img);
            allkeys.add(pyramidDenseSIFT.getByteKeypoints());//0.005f
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
            if(BLOCKSPATIAL) {
                BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(
                        bovw, BLOCKX, BLOCKY); //2 2
                return spatial.aggregate(pdsift.getByteKeypoints(ENERGYTHRESH2), fImage.getBounds()).normaliseFV();//0.015f
            }
            else{
                PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(
                        bovw, BLOCKX, BLOCKY); //2 2
                return spatial.aggregate(pdsift.getByteKeypoints(ENERGYTHRESH2), fImage.getBounds()).normaliseFV();//0.015f
            }
        }
    }
}