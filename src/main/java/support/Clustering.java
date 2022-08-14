package support;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.clustering.hdbscan.HdbscanTrainer;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.response.EmptyResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;

import java.awt.*;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

import static support.Plotting.getNewXYChart;

public class Clustering {
    public static void main(String[] args) throws IOException {
        // Create a new CSVDataReader object to read the 2d data from
        var reader = new CSVDataReader("C:\\Users\\zenith\\Documents\\MyDatasets\\simple-2d-data-train.csv");

        // Get the dataset from the 2d data in the CSVDataReader
        var dataset = reader.generate2dDataDouble("Feature1", "Feature2");

        // Create the clustering trainer and model
        // - the min points in a cluster to be considered a cluster is set to 5
        var trainer = new HdbscanTrainer(5);
        var model = trainer.train(dataset);

        // Get the info about the clusters
        var clusterLabels = model.getClusterLabels();
        var labelsSet = new HashSet<>(clusterLabels);
        System.out.println(labelsSet);

        // Put the MutableDataset object into a x/y list, to then graph.
        List<Double> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();
        Plotting.setXandYListsFromDataset(xList, yList, dataset);


        // Create a chart showing the basic 2d data
        XYChart chart = Plotting.getNewXYChart("Dataset");
        Plotting.addSeriesToChart(chart, xList, yList, "Points", Color.blue, SeriesMarkers.CIRCLE);
        Plotting.displayChart(chart);


        // Sort the data into individual lists with keys referring to the cluster each value is in
        Map<Integer, List<Double>> mapX = new HashMap<>();
        Map<Integer, List<Double>> mapY = new HashMap<>();
        int i = 0;
        for (Integer label : clusterLabels) {
            List<Double> lx = mapX.computeIfAbsent(label, p -> new ArrayList<>());
            lx.add(xList.get(i));
            List<Double> ly = mapY.computeIfAbsent(label, p -> new ArrayList<>());
            ly.add(yList.get(i));
            i++;
        }

        // Plot the colored clusters
        // Since there are 3 distict clusters, we add 3 different colors to a queue to be used in the chart.
        Queue<Color> colors = new ArrayDeque<>();
        colors.add(Color.cyan); colors.add(Color.green); colors.add(Color.magenta);

        chart = getNewXYChart("Cluster Result");
        Plotting.addAllSeriesToChart(chart,  mapX,  mapY, colors);

        // Get outlier information about one of the points
        System.out.println(model.getOutlierScores().get(0));

        // Load the prediction data
        var predictData = new CSVDataReader("C:\\Users\\zenith\\Documents\\MyDatasets\\simple-2d-data-predict.csv");
        var predictDataset = predictData.generate2dDataDouble("Feature1","Feature2");

        // Arrange the prediction data into two lists
        xList = new ArrayList<>();
        yList = new ArrayList<>();
        Plotting.setXandYListsFromDataset(xList, yList, predictDataset);

        // Create the predictions on the predictDataset
        var predictions = model.predict(predictDataset);

        // Create the prediction labels and sort the points based on the labels
        List<Integer> predictionLabels = new ArrayList<>();
        predictions.forEach(p -> predictionLabels.add(p.getOutput().getID()));

        i = 0;
        for (Integer label : predictionLabels) {
            List<Double> lx = mapX.computeIfAbsent(label, p -> new ArrayList<>());
            lx.add(xList.get(i));
            List<Double> ly = mapY.computeIfAbsent(label, p -> new ArrayList<>());
            ly.add(yList.get(i));
            i++;
        }

        // Plot the new data w/ the clustering color-coded
        colors.add(Color.cyan); colors.add(Color.green); colors.add(Color.magenta);
        chart = getNewXYChart("Prediction Result");
        Plotting.addAllSeriesToChart(chart,  mapX,  mapY, colors);
        Plotting.displayChart(chart);

        System.out.println("The outlier score for (4.25, 1.5) is : " + predictions.get(0).getOutput().getScore());
        System.out.println("The outlier score for (1.5, 3.0) is : " + predictions.get(2).getOutput().getScore());


    }
}
