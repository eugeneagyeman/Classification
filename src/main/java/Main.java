import java.io.IOException;

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
