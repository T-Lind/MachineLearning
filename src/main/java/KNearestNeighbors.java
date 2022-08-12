import org.tribuo.*;
import org.tribuo.clustering.evaluation.*;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.clustering.kmeans.KMeansTrainer;
import org.tribuo.util.Util;

/**
 * An example class to evaluate the K nearest neighbors
 */
public class KNearestNeighbors {
    public static void main(String[] args){
        System.out.println("Hello world");

        var eval = new ClusteringEvaluator();

        var data = new MutableDataset<>(new GaussianClusterDataSource(500, 1L));
        var test = new MutableDataset<>(new GaussianClusterDataSource(500, 2L));

        var trainer = new KMeansTrainer(5, /* centroids */
                10, /* iterations */
                KMeansTrainer.Distance.EUCLIDEAN, /* distance function */
                1, /* number of compute threads */
                42 /* RNG seed */
        );

        var startTime = System.currentTimeMillis();
        var model = trainer.train(data);
        var endTime = System.currentTimeMillis();
        System.out.println("Training with 5 clusters took " + Util.formatDuration(startTime,endTime));

        var centroids = model.getCentroids();
        for (var centroid : centroids) {
            System.out.println(centroid);
        }


    }
}
