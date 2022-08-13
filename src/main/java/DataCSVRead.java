import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.data.columnar.FieldExtractor;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.extractors.DoubleExtractor;
import org.tribuo.data.columnar.extractors.FloatExtractor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.math.optimisers.SGD;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.sgd.objectives.SquaredLoss;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer;
import org.tribuo.util.Util;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * A runner class to experiment with loading data. Loads an external CSV file of wine data.
 */
public class DataCSVRead {
    public static void main(String[] args) throws IOException {
        // Get the entire dataset
        //var csvPath = Paths.get("C:\\Users\\zenith\\Downloads\\wine-quality-master\\wine-quality-master\\winequality\\winequality-red.csv");
        var csvPath = Paths.get("C:\\Users\\zenith\\Downloads\\data.csv");


        // Read all lines from the dataset
        var csvLines = Files.readAllLines(csvPath, StandardCharsets.UTF_8);


        var responseProcessor = new FieldResponseProcessor("data","UNK",new LabelFactory());

        var fieldProcessors = new HashMap<String, FieldProcessor>();
        fieldProcessors.put("data", new DoubleFieldProcessor("data"));

        var rowProcessor = new RowProcessor<Label>(responseProcessor, fieldProcessors);


        var csvSource = new CSVDataSource<Label>(csvPath,rowProcessor,true);
        var datasetFromCSV = new MutableDataset<Label>(csvSource);

        System.out.println(datasetFromCSV.getExample(1));
    }

    public static void printExample(Example<Label> e) {
        System.out.println("Output = " + e.getOutput().toString());
        System.out.println("Metadata = " + e.getMetadata());
        System.out.println("Weight = " + e.getWeight());
        System.out.println("Features = [" + StreamSupport.stream(e.spliterator(), false).map(Feature::toString).collect(Collectors.joining(",")) + "]");
    }
}
