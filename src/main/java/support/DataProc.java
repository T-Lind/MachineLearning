package support;

import org.tensorflow.Graph;
import org.tensorflow.framework.initializers.Glorot;
import org.tensorflow.framework.initializers.VarianceScaling;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.nn.Conv2d;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.types.TFloat32;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.interop.tensorflow.example.GraphDefTuple;
import org.tribuo.util.Util;

import java.util.Arrays;

/**
 * Helpful data processing features
 */
public final strictfp class DataProc {
    /**
     * Path to my dataset folder
     */
    public static final String DATASETS = "C:\\Users\\zenith\\Documents\\MyDatasets\\";

    /**
     * Convert from a DataSource to a MutableDataset.
     * @param source the input datasource (Probably a csvDataSource)
     * @return the MutableDataset version of the same data
     */
    public static Dataset convertSourceSet(DataSource<Label> source){
        return new MutableDataset<>(source);
    }

    public static Model train(Dataset dataset, Trainer trainer){
        var linStartTime = System.currentTimeMillis();
        var linModel = trainer.train(dataset);
        var linEndTime = System.currentTimeMillis();
        System.out.println();
        System.out.println(trainer.getClass() + " model training took " + Util.formatDuration(linStartTime,linEndTime));

        return linModel;
    }

    /**
     * Evaluate a model on a test set
     * @param testset the set to test the model on
     * @param model the model to test on the data
     * @param eval the evaluator
     * @return the built evaluator
     */
    public static Evaluation evaluate(MutableDataset testset, Model model, Evaluator eval){
        var startTime = System.currentTimeMillis();
        var evaluation = eval.evaluate(model, testset);
        var endTime = System.currentTimeMillis();

        System.out.println();
        System.out.println(model.getClass() + " model evaluation took " + Util.formatDuration(startTime,endTime));
        System.out.println(evaluation);

        return evaluation;
    }

//    public static GraphDefTuple generateCNN(String inputName){
//        final String PADDING_TYPE = "SAME";
//        var graph = new Graph();
//
//        Ops tf = Ops.create(graph);
//
//        Glorot<TFloat32> initializer = new Glorot<>(VarianceScaling.Distribution.TRUNCATED_NORMAL, 42);
//
//        Placeholder<TFloat32> input = tf.withName(inputName).placeholder(TFloat32.class, Placeholder.shape(Shape.of(2,2)));
//
//        Variable<TFloat32> conv1Weights = tf.variable(initializer.call(tf, tf.array(5L, 5, 1, 32), TFloat32.class));
//        Conv2d<TFloat32> conv1 = tf.nn.conv2d()
//        Variable<TFloat32> conv1Biases = tf.variable(tf.fill(tf.array(32), tf.constant(0.0f)));
//        Relu<TFloat32> relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases));
//
//        return null;
//    }
}
