package support;


import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.TextFieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class CSVDataReader{
    private Path csvPath;

    private List<String> csvLines;

    private HashMap<String, MutableDataset<Label>> generatedData;

    /**
     * Set up the reader and read the CSV lines
     * @param csvPath the path to the CSV file
     * @throws IOException from the readAllLines method call
     */
    public CSVDataReader(String csvPath) throws IOException {
        this.csvPath = Paths.get(csvPath);
        csvLines = Files.readAllLines(this.csvPath, StandardCharsets.UTF_8);

        generatedData = new HashMap<>();
    }

    /**
     * Load data from a specific column in your CSV file.
     * @param name the name of the column being read
     * @param dataType the type of data being read. Use Double.class or String.class
     * @return <h2 style="background-color:White;">THE CASTED DATASET TO USE BROADLY!</h2>
     */
    public Dataset generateDataColumn(String name, Object dataType){
        var responseProcessor = new FieldResponseProcessor(name,"UNK",new LabelFactory());

        FieldProcessor processor;


        if(dataType == Double.class){
            processor = new DoubleFieldProcessor(name);
        }
        else if(dataType == String.class){
            var textPipeline = new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2);
            processor = new TextFieldProcessor(name, textPipeline);
        }
        else{
            throw new RuntimeException("Improper dataType specified in CSVDataReader.generateDataColumn()");
        }

        var fieldProcessors = new HashMap<String, FieldProcessor>();
        fieldProcessors.put(name, processor);

        var rowProcessor = new RowProcessor<Label>(responseProcessor, fieldProcessors);

        var csvSource = new CSVDataSource<Label>(csvPath,rowProcessor,true);

        return new MutableDataset<>(csvSource);
    }


    public MutableDataset<Label> getMutableDataset(String name){
        return generatedData.get(name);
    }

    public Label getData(String name, int index){
        return getOutput(getMutableDataset(name).getExample(index));
    }


    private Label getOutput(Example<Label> e) {
        return e.getOutput();
    }


}
