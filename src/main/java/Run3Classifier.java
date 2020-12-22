import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
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
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
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
    private static final String CACHE_PATH = "src/main/java/com/ea4u16/ch12/cache";
    private static final Timer timer = new Timer();
    static  {
        System.out.println("Running Classifier...");
        try {
            //Grouped Dataset
            GroupedDataset<String, VFSListDataset<Caltech101.Record<FImage>>, Caltech101.Record<FImage>> allData =
                    Caltech101.getData(ImageUtilities.FIMAGE_READER);

            GroupedDataset<String, ListDataset<Caltech101.Record<FImage>>, Caltech101.Record<FImage>> data =
                    GroupSampler.sample(allData, 5, false);

            GroupedRandomSplitter<String, Caltech101.Record<FImage>> splits =
                    new GroupedRandomSplitter<String, Caltech101.Record<FImage>>(data, 15, 0, 15);

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
                assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pyramidDenseSIFT);
                cacheFile = new File(CACHE_PATH);
                cacheFile.createNewFile();
                IOUtils.writeToFile(assigner, cacheFile);
            } catch (IOException noFile) {
                System.out.println("Error in accessing cache... creating new cache");
                assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pyramidDenseSIFT);
                cacheFile = new File(CACHE_PATH);
                cacheFile.createNewFile();
                IOUtils.writeToFile(assigner, cacheFile);
            }

            FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> extractor = new PHOWExtractor(pyramidDenseSIFT, assigner);

            //Exercise 1: Homogeneous Kernel Map
            HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
            FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> homogeneousFeatureExtractor = homogeneousKernelMap.createWrappedExtractor(extractor);

            //Exercise 2: Feature Caching
            DiskCachingFeatureExtractor<DoubleFV, Caltech101.Record<FImage>> diskCachingFeatureExtractor = new DiskCachingFeatureExtractor<>(cacheFile, homogeneousFeatureExtractor);
            LiblinearAnnotator<Caltech101.Record<FImage>, String> ann = new LiblinearAnnotator<>(
                    diskCachingFeatureExtractor,
                    LiblinearAnnotator.Mode.MULTICLASS,
                    SolverType.L2R_L2LOSS_SVC,
                    1.0,
                    0.00001);

            timer.start();
            ann.train(splits.getTrainingDataset());
            timer.stop();

            long resultantTime = timer.duration();
            System.out.println("Time: " + resultantTime);

            ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>> eval =
                    new ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>>(
                            ann,
                            splits.getTestDataset(),
                            new CMAnalyser<Caltech101.Record<FImage>, String>(CMAnalyser.Strategy.SINGLE));
            Map<Caltech101.Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
            CMResult<String> result = eval.analyse(guesses);
            result.getDetailReport();
            System.out.println("CH12 Completed.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<Caltech101.Record<FImage>> sample, PyramidDenseSIFT<FImage> pyramidDenseSIFT) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new
                ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
        for (Caltech101.Record<FImage> rec : sample) {
            FImage img = rec.getImage();
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

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair>
                assigner) {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(Caltech101.Record<FImage> object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<>(
                    bovw, 2, 2);
            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}
