import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.clustering.kmeans.KMeansTrainer;
import org.tribuo.util.Util;
import support.CSVDataReader;

import java.io.IOException;

public class MyKNearestNeighbors {
    public static void main(String[] args) throws IOException {
        CSVDataReader reader = new CSVDataReader("C:\\Users\\zenith\\Documents\\MyDatasets\\random_distro_two_centroids.csv");
        var data = reader.generateDataColumn("values", Double.class);

        var trainer = new KMeansTrainer(
                2,
                100,
                KMeansTrainer.Distance.EUCLIDEAN,
                1,
                42
        );

        var startTime = System.currentTimeMillis();
        var model = trainer.train(data);
        var endTime = System.currentTimeMillis();
        System.out.println("Training with 2 clusters took " + Util.formatDuration(startTime,endTime));

        var centroids = model.getCentroids();
        for (var centroid : centroids) {
            System.out.println(centroid);
        }
    }
}
