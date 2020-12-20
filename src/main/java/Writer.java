import org.apache.commons.vfs2.FileObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

public class Writer {

    FileWriter writer;
    String path;

    public Writer(int runNo) {
        path = "run" + runNo + ".txt";
        try {
            writer = new FileWriter(path);
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
        return writeStr(imageFile.getName().getBaseName()+" "+classifier+"\n");
    }

    public int writeResults(Map<FileObject, String> results) {
        final int[] r = {1};
        results.forEach( (fo, c) -> {
            r[0] &=
                    writeResult(fo, c);
        });

        return r[0]-1;
    }

    public int writeStr(String str) {
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
