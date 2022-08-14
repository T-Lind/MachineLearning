package support;


import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.TextFieldProcessor;
import org.tribuo.data.columnar.processors.response.EmptyResponseProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * A custom wrapper class to ease reading data from a csv file!
 * Specify a path and you can generate the dataset objects for each column.
 */
public strictfp class CSVDataReader{
    /**
     * The file path object to the csv file
     */
    private transient Path csvPath;

    /**
     * A hashmap to store the generated datasets and access them later
     */
    private transient HashMap<String, MutableDataset<Label>> generatedData;
    private transient HashMap<String, DataSource<Label>> generatedSources;

    /**
     * Set up the reader and read the CSV lines
     * @param csvPath the path to the CSV file
     * @throws IOException from the readAllLines method call
     */
    public CSVDataReader(String csvPath) throws IOException {
        this.csvPath = Paths.get(csvPath);

        generatedData = new HashMap<>();
        generatedSources = new HashMap<>();
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
        generatedSources.put(name, csvSource);


        return new MutableDataset<>(csvSource);
    }

    /**
     * Generate data from a CSV file with 2d data
     * @param x the x label
     * @param y the y label
     * @return the MutableDataset containing the data
     */
    public MutableDataset<ClusterID> generate2dDataDouble(String x, String y){
        Map<String, FieldProcessor> regexMappingProcessors = new HashMap<>();

        regexMappingProcessors.put(x, new DoubleFieldProcessor(x));
        regexMappingProcessors.put(y, new DoubleFieldProcessor(y));

        RowProcessor<ClusterID> rowProcessor = new RowProcessor<>(new EmptyResponseProcessor<>(new ClusteringFactory()), regexMappingProcessors);

        var csvDataSource = new CSVDataSource<>(csvPath, rowProcessor, false);
        var dataset = new MutableDataset<>(csvDataSource);

        return dataset;
    }

    /**
     * Get the mutable <b>(NON CASTED, UNLIKE WHAT THE GENERATE METHOD RETURNS)</b> column dataset from the hashmap
     * @param name is the name of the column dataset
     * @return the column mutable dataset
     */
    public MutableDataset<Label> getMutableDataset(String name){
        return generatedData.get(name);
    }

    /**
     * Get the dataSource object for a certain column/generatedSource
     * @param name the name to get
     * @return the datasource object for name
     */
    public DataSource<Label> getDataSource(String name){
        return generatedSources.get(name);
    }

    /**
     * Get a specific value from a column at a specific index
     * @param name the name of the column
     * @param index the index in the column (row, 0=first)
     * @return the Label value of whatever's in there.
     */
    public Label getData(String name, int index){
        return getOutput(getMutableDataset(name).getExample(index));
    }

    /**
     * Split the test data and training data
     * @param name the name of the data
     * @param splitRatio the ratio to split the data at
     * @param seed the random seed to use when splitting
     * @return the TrainTestSplitter object
     */
    public TrainTestSplitter splitData(String name, double splitRatio, int seed){
        return new TrainTestSplitter<>(getDataSource(name), splitRatio, seed);
    }

    /**
     * Get the output of an Example object
     * @param e the example object
     * @return the Label object from the input Example
     */
    private Label getOutput(Example<Label> e) {
        return e.getOutput();
    }

    /**
     * Convert from a DataSource to a MutableDataset.
     * @param source the input datasource (Probably a csvDataSource)
     * @return the MutableDataset version of the same data
     */
    public Dataset convertSourceSet(DataSource<Label> source){
        return new MutableDataset<>(source);
    }
}
