import ml.dmlc.xgboost4j.java.Rabit;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import support.CSVDataReader;

import java.io.IOException;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
/**
 * A runner class to experiment with loading data. Loads an external CSV file of wine data.
 */
public class DataCSVRead {
    public static void main(String[] args) throws IOException {

        CSVDataReader reader = new CSVDataReader("C:\\Users\\zenith\\Downloads\\data.csv");
        reader.generateDataColumn("data", Double.class);

        var data = reader.getDataColumn("data");

        System.out.println(reader.getData("data", 1));
    }

    public static void printExample(Example<Label> e) {
        System.out.println("Output = " + e.getOutput().toString());
        System.out.println("Metadata = " + e.getMetadata());
        System.out.println("Weight = " + e.getWeight());
        System.out.println("Features = [" + StreamSupport.stream(e.spliterator(), false).map(Feature::toString).collect(Collectors.joining(",")) + "]");
    }
}
