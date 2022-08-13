package support;

import java.awt.Color;
import org.knowm.xchart.style.markers.Marker;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.clustering.ClusterID;

import java.util.List;
import java.util.Map;
import java.util.Queue;

public class Plotting {
    /**
     * A method to get a new instance of a chart, configured the same way each time
     * @param title the title of the chart
    */
     public static XYChart getNewXYChart(String title) {
        XYChart chart = new XYChartBuilder().width(600).height(400).title(title).xAxisTitle("X").yAxisTitle("Y").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setChartTitleVisible(false);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setMarkerSize(8);
        chart.getStyler().setPlotGridHorizontalLinesVisible(false);
        chart.getStyler().setPlotGridVerticalLinesVisible(false);
        return chart;
    }

    // A method to add a set of (x,y) points to a chart
    public static void addSeriesToChart(XYChart chart, List<Double> xList, List<Double> yList, String seriesName, Color color, Marker marker) {
        XYSeries xYseries = chart.addSeries(seriesName,
                xList.stream().mapToDouble(Double::doubleValue).toArray(),
                yList.stream().mapToDouble(Double::doubleValue).toArray());
        xYseries.setMarkerColor(color);
        xYseries.setMarker(marker);
    }

    // A method to multiple sets of (x,y) points to a chart
    public static void addAllSeriesToChart(XYChart chart, Map<Integer, List<Double>> mapX, Map<Integer, List<Double>> mapY, Queue<Color> colors) {
        for (Map.Entry<Integer, List<Double>> entry : mapX.entrySet()) {
            if (entry.getKey() == 0) {    // The Outlier label
                List<Double> xList = entry.getValue();
                List<Double> yList = mapY.get(0);
                addSeriesToChart(chart, xList, yList, "Points" + entry.getKey(), Color.darkGray, SeriesMarkers.CIRCLE);
            } else {                      // Valid Cluster labels
                List<Double> xList = entry.getValue();
                List<Double> yList = mapY.get(entry.getKey());
                addSeriesToChart(chart, xList, yList, "Points" + entry.getKey(), colors.poll(), SeriesMarkers.CIRCLE);
            }
        }
    }



    // A method which extracts the (x,y) points from the dataset
    public static void setXandYListsFromDataset(List<Double> xList, List<Double> yList, Dataset<ClusterID> dataset) {
        for (Example<ClusterID> ex : dataset) {
            int i = 0;
            for (Feature f : ex) {
                if (i == 0)  xList.add(f.getValue());
                if (i == 1)  yList.add(f.getValue());
                i++;
            }
        }
    }
}
