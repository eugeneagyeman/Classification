import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.feature.FImage2DoubleFV;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

/** To do list:
 *  > create 8x8 patches, sample every 4 pixels in x,y direction -> 2x2 grid where each square has 16 pixels?
 *  > Take pixels from patches and potentially mean centre and normalise (working with intensities, not gradient-magnitude)
 *  > flatten each patch into a vector
 *  > map each patch to a representative visual word via K means algorithm
 *  > build classifier that takes into account number of visual words in the image and classify accordingly
 * Key info:
 * > Vocab / codebook size , try 500 as the size of the codebook
 * > Use LiblinearAnnotator
 *
 */
public class Run2 {

    FeatureExtractor fe = new FImage2DoubleFV();
    LiblinearAnnotator ll = new LiblinearAnnotator(fe, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L1R_L2LOSS_SVC, 1, 1);



}
