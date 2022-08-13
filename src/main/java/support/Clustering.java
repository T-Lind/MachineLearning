package support;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.response.EmptyResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;

import java.awt.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Clustering {
    public static void main(String[] args) {


        Map<String, FieldProcessor> regexMappingProcessors = new HashMap<>();
        regexMappingProcessors.put("Feature1", new DoubleFieldProcessor("Feature1"));
        regexMappingProcessors.put("Feature2", new DoubleFieldProcessor("Feature2"));
        RowProcessor<ClusterID> rowProcessor = new RowProcessor<>(new EmptyResponseProcessor<>(new ClusteringFactory()), regexMappingProcessors);

        var csvDataSource = new CSVDataSource<>(Paths.get("C:\\Users\\zenith\\Documents\\MyDatasets\\2d_data_small.csv"), rowProcessor, false);
        var dataset = new MutableDataset<>(csvDataSource);



        List<Double> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();
        Plotting.setXandYListsFromDataset(xList, yList, dataset);

        var chart = Plotting.getNewXYChart("Dataset");
        Plotting.addSeriesToChart(chart, xList, yList, "Points", Color.blue, SeriesMarkers.CIRCLE);
        BitmapEncoder.getBufferedImage(chart);


    }
}
