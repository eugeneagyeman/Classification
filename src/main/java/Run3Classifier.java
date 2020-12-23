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

    //TODO:
    static  {
        System.out.println("Running Classifier...");
        try {

            VFSGroupDataset<FImage> trainImages =
                    new VFSGroupDataset<>("/Users/eugeneagyeman/Documents/CSP3/Computer Vision/CW3/classification/training", ImageUtilities.FIMAGE_READER);

            VFSGroupDataset<FImage> testImages = new VFSGroupDataset<FImage>("/Users/eugeneagyeman/Documents/CSP3/Computer Vision/CW3/classification/testing",ImageUtilities.FIMAGE_READER);
            //Debug to show images
            /*for(final Map.Entry<String,VFSListDataset<FImage>> entry: trainImages.entrySet()) {
                DisplayUtilities.display(entry.getKey(),entry.getValue());
            }*/

            /*
            GroupedDataset<String, ListDataset<FImage>, FImage> data =
                    GroupSampler.sample(images, 15, false);

            GroupedRandomSplitter<String, FImage> splits =
                    new GroupedRandomSplitter<String, FImage>(data, 15, 0, 15);*/

            //SIFTs
            DenseSIFT denseSIFT = new DenseSIFT(5, 7);
            PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<>(denseSIFT, 6f, 7);


            //Assigner
            HardAssigner<byte[], float[], IntFloatPair> assigner = null;
            File cacheFile;
            try {
                cacheFile = new File(CACHE_PATH);
                assigner = IOUtils.readFromFile(cacheFile);
                System.out.println("Cache Loaded...");
            } catch (NullPointerException noFile) {
                System.out.println("Cache Not Found, creating new one...");
                assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainImages, 30), pyramidDenseSIFT);
                cacheFile = new File(CACHE_PATH);
                cacheFile.createNewFile();
                IOUtils.writeToFile(assigner, cacheFile);
            } catch (IOException noFile) {
                System.out.println("Error in accessing cache... creating new cache");
                assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainImages, 30), pyramidDenseSIFT);
                cacheFile = new File(CACHE_PATH);
                cacheFile.createNewFile();
                IOUtils.writeToFile(assigner, cacheFile);
            }

            FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pyramidDenseSIFT, assigner);

            //Exercise 2: Feature Caching
            //DiskCachingFeatureExtractor<DoubleFV, FImage> diskCachingFeatureExtractor = new DiskCachingFeatureExtractor<>(cacheFile, homogeneousFeatureExtractor);

            /*LiblinearAnnotator<Caltech101.Record<FImage>, String> ann = new LiblinearAnnotator<>(
                    diskCachingFeatureExtractor,
                    LiblinearAnnotator.Mode.MULTICLASS,
                    SolverType.L2R_L2LOSS_SVC,
                    1.0,
                    0.00001);*/

            NaiveBayesAnnotator<FImage,String> svmann = new NaiveBayesAnnotator<FImage, String>(extractor, NaiveBayesAnnotator.Mode.ALL);

            timer.start();
            svmann.train(trainImages);
            timer.stop();

            long resultantTime = timer.duration();
            System.out.println("Time: " + resultantTime);

            ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(svmann,testImages,new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

            Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
            CMResult<String> result = eval.analyse(guesses);
            result.getDetailReport();
            System.out.println("Classification Completed.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pyramidDenseSIFT) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new
                ArrayList<>();
        for (FImage img : sample) {
            pyramidDenseSIFT.analyseImage(img);
            allkeys.add(pyramidDenseSIFT.getByteKeypoints(0.005f));
        }
        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }
    class Record<FImage> {
        private String imageClass;
        private FImage image;

        public Record(FImage image) {
        }
        FImage getImage() {
            return image;
        }
        String getImageClass() {
            return imageClass;
        }
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
            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), fImage.getBounds()).normaliseFV();
        }
    }
}
