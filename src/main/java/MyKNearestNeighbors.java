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
        // Create a reader object and feed it the csv path
        CSVDataReader reader = new CSVDataReader("C:\\Users\\zenith\\Documents\\MyDatasets\\random_distro_two_centroids.csv");

        // Load the "values" column and store the CASTED Dataset object into data.
        var data = reader.generateDataColumn("values", Double.class);

        // Create the KMeansTrainer - note that K++ is being used
        var trainer = new KMeansTrainer(
                2,
                100,
                KMeansTrainer.Distance.EUCLIDEAN,
                KMeansTrainer.Initialisation.PLUSPLUS,
                1,
                42
        );

        // Measure training time
        var startTime = System.currentTimeMillis();
        var model = trainer.train(data);
        var endTime = System.currentTimeMillis();
        System.out.println("Training with 2 clusters took " + Util.formatDuration(startTime,endTime));

        // Print the centroids
        var centroids = model.getCentroids();
        for (var centroid : centroids) {
            System.out.println(centroid);
        }
    }
}
