
/** To do list:
 *  > Find interest points in the image -> harris stephens detector? DoGE ?
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


}
