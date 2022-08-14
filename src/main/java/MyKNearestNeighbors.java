import org.tribuo.clustering.kmeans.KMeansTrainer;
import org.tribuo.util.Util;
import support.DataProc;
import support.DataReader;

import java.io.IOException;

public class MyKNearestNeighbors {
    public static void main(String[] args) throws IOException {
        // Create a reader object and feed it the csv path
        DataReader reader = new DataReader("C:\\Users\\zenith\\Documents\\MyDatasets\\random_distro_two_centroids.csv");

        // Load the "values" column and store the CASTED Dataset object into data.
        var data = reader.generateDataColumn("values", Double.class);

        var splitter = reader.splitData("values", 0.7, 42);


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
        var model = trainer.train(DataProc.convertSourceSet(splitter.getTrain()));
        var endTime = System.currentTimeMillis();
        System.out.println("Training with 2 clusters took " + Util.formatDuration(startTime,endTime));

        // Print the centroids
        var centroids = model.getCentroids();
        for (var centroid : centroids) {
            System.out.println(centroid);
        }

        // Evaluate the model
//        var eval = new ClusteringEvaluator();
//
//        var trainEvaluation = eval.evaluate(model, reader.getMutableDataset("values"));
//        System.out.println(trainEvaluation.toString());
    }
}
