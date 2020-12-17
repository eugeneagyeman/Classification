import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Writer {

    FileWriter writer;

    public Writer(String path) {
        File f = new File(path);
        try {
            writer = new FileWriter(path);
        } catch (IOException e) {
        }
    }

    public int write(String str) {
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

    public static void main(String[] args) {
        Writer w = new Writer("file.txt");
        System.out.println(w.write("dnlk"));
        System.out.println(w.write("\tkjrgnl"));
        System.out.println(w.write("kjrgnl\n"));
        System.out.println(w.write("kjrgnl\n"));
        System.out.println(w.flush());
    }

}
