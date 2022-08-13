package support;


import ml.dmlc.xgboost4j.java.Rabit;
import org.tensorflow.proto.framework.DataType;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class CSVDataReader {
    private Path csvPath;

    private List<String> csvLines;

    private HashMap<String, MutableDataset<Label>> generatedData;

    public CSVDataReader(String csvPath) throws IOException {
        this.csvPath = Paths.get(csvPath);
        csvLines = Files.readAllLines(this.csvPath, StandardCharsets.UTF_8);

        generatedData = new HashMap<>();
    }

    public void generateDataColumn(String name, Object dataType){
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

        generatedData.put(name, new MutableDataset<Label>(csvSource));
    }


    public MutableDataset<Label> getDataColumn(String name){
        return generatedData.get(name);
    }

    public Label getData(String columnName, int index){
        return getOutput(getDataColumn(columnName).getExample(index));
    }

    private Label getOutput(Example<Label> e) {
        return e.getOutput();
    }
}
