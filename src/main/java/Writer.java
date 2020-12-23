import org.apache.commons.vfs2.FileObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

/**
 * Simple helper class used to assist in writing results of classification algorithms to text files.
 *
 * Written by Emily James, Dec 2020
 */
public class Writer {

    FileWriter writer;
    String path;

    /**
     * Automatically creates a file in the working directory under the name
     * <tt>run[runNo].txt</tt>.
     * @param runNo
     */
    public Writer(int runNo) {
        path = "run" + runNo + ".txt";
        try {
            writer = new FileWriter(path);
        } catch (IOException e) {
        }
    }

    /**
     * Initialise a Writer on a specific file path. Only reccomended to use for debugging.
     * @param fn
     */
    public Writer(String fn) {
        try {
            writer = new FileWriter(fn);
        } catch (IOException e) {
        }
    }

    public int reopenFile() {
        try {
            writer = new FileWriter(path);
            return 0;
        } catch (IOException e) {
            return -1;
        }
    }

    public int writeResult(FileObject imageFile, String classifier) {
        return writeResult(imageFile.getName().getBaseName(), classifier);
    }

    public int writeResult(String imageFileName, String classifier) {
        return writeStr(imageFileName+" "+classifier+"\n");
    }

    /**
     * A variant to {@code writeResults(..)} which uses FileObjects instead
     * of Strings.
     * @param results
     * @return
     */
    public int writeFOResults(Map<FileObject, String> results) {
        final int[] r = {1};
        results.forEach( (fo, c) -> {
            r[0] &=
                    writeResult(fo, c);
        });

        return r[0]-1;
    }

    public int writeResults(Map<String, String> results) {
        final int[] r = {1};
        results.forEach( (fo, c) -> {
            r[0] &=
                    writeResult(fo, c);
        });

        return r[0]-1;
    }


    private int writeStr(String str) {
        try {
            writer.write(str);
            return 0;
        } catch (IOException e) {
            return -1;
        }
    }

    public int flush() {
        try {
            writer.flush();
            return 0;
        } catch (IOException e) {
            return -1;
        }
    }

    public FileWriter getWriter(){
        return writer;
    }

    public int closeFile() {
        try {
            writer.close();
            return 0;
        } catch (IOException e) {
            return -1;
        }
    }

}
