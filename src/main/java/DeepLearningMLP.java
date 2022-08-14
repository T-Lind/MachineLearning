import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import org.tribuo.*;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.interop.tensorflow.*;
import org.tribuo.interop.tensorflow.example.*;
import org.tribuo.regression.*;
import org.tribuo.regression.evaluation.*;
import org.tribuo.util.Util;

import org.tensorflow.*;
import org.tensorflow.framework.initializers.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.*;
import org.tensorflow.op.core.*;
import org.tensorflow.types.*;
import support.DataProc;

/**
 * Build a neural network to recognize numbers in the MINST dataset
 */
public class DeepLearningMLP {
    public static void main(String[] args) throws IOException {
        // load MNIST
        var labelFactory = new LabelFactory();
        var labelEval = new LabelEvaluator();
        var mnistTrainSource = new IDXDataSource<>(Paths.get(DataProc.DATASETS+"mnist\\train-images-idx3-ubyte.gz"),Paths.get(DataProc.DATASETS+"mnist\\train-labels-idx1-ubyte.gz"),labelFactory);
        var mnistTestSource = new IDXDataSource<>(Paths.get(DataProc.DATASETS+"mnist\\t10k-images-idx3-ubyte.gz"),Paths.get(DataProc.DATASETS+"mnist\\t10k-labels-idx1-ubyte.gz"),labelFactory);
        var mnistTrain = new MutableDataset<>(mnistTrainSource);
        var mnistTest = new MutableDataset<>(mnistTestSource);

        // Build classification model - multilayer perceptron is MLP, which is also known as a feedforward ANN
        var mnistInputName = "MNIST_INPUT";
        var mnistMLPTuple = MLPExamples.buildMLPGraph(
                mnistInputName, // The input placeholder name
                mnistTrain.getFeatureMap().size(), // The number of input features
                new int[]{300,200,30}, // The hidden layer sizes - 3 hidden layers
                mnistTrain.getOutputs().size() // The number of output labels - 10, 1 for each number
        );

        // Create data converters
        var mnistDenseConverter = new DenseFeatureConverter(mnistInputName);
        var mnistOutputConverter = new LabelConverter();

        // Define the parameters of the MLP training algorithm
        var gradAlgorithm = GradientOptimiser.ADAGRAD;
        var gradParams = Map.of("learningRate",0.3f,"initialAccumulatorValue",0.01f);

        // Create the trainer
        var mnistMLPTrainer = new TensorFlowTrainer<Label>(mnistMLPTuple.graphDef,
                mnistMLPTuple.outputName, // the name of the logit operation
                gradAlgorithm,            // the gradient descent algorithm
                gradParams,               // the gradient descent hyperparameters
                mnistDenseConverter,      // the input feature converter
                mnistOutputConverter,     // the output label converter
                16,  // training batch size
                10,  // number of training epochs
                16,  // test batch size of the trained model
                -1   // disable logging of the loss value
        );

        var mlpStart = System.currentTimeMillis();
        var mlpModel = mnistMLPTrainer.train(mnistTrain);
        var mlpEnd = System.currentTimeMillis();
        System.out.println("MNIST MLP training took " + Util.formatDuration(mlpStart,mlpEnd));

        var mlpEvaluation = labelEval.evaluate(mlpModel,mnistTest);
        System.out.println(mlpEvaluation.toString());
        System.out.println(mlpEvaluation.getConfusionMatrix().toString());
    }
}
