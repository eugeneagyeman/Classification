import de.bwaldvogel.liblinear.SolverType;
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
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.IOException;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws IOException {

        /*// have to specify a caltech record rather than the abstract record made by java
        VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>("zip:training.zip", ImageUtilities.FIMAGE_READER);


        DenseSIFT dsift = new DenseSIFT(3, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 4,6,8,10);

        // I think this works..? seemed to have worse accuracy however
        HardAssigner<byte[], float[], IntFloatPair> assigner;
        assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingSet, 30), pdsift);

        FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> extractor = new PHOWExtractor(pdsift, assigner);
        // using homogenous kernal map... accuracy went up when using it
        HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> hkmExtractor = hkm.createWrappedExtractor(extractor);


        LiblinearAnnotator<Caltech101.Record<FImage>, String> ann = new LiblinearAnnotator<>(
                hkmExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, .00001);
        ann.train(splits.getTrainingDataset());

        ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>> eval =
                new ClassificationEvaluator<>(
                        ann, splits.getTestDataset(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

        Map<Caltech101.Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        // print results to sdtout
        System.out.println(result.getDetailReport());*/
        new Run3Classifier();
    }


}
