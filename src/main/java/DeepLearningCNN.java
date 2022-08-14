import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.interop.tensorflow.*;
import org.tribuo.interop.tensorflow.example.CNNExamples;
import org.tribuo.interop.tensorflow.example.MLPExamples;
import org.tribuo.util.Util;
import support.DataProc;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;

public class DeepLearningCNN {
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

        // Create data converters
        var mnistOutputConverter = new LabelConverter();

        // Define the parameters of the MLP training algorithm
        var gradAlgorithm = GradientOptimiser.ADAGRAD;
        var gradParams = Map.of("learningRate",0.1f,"initialAccumulatorValue",0.01f);

        // Create the CNN object info
        var mnistCNNTuple = CNNExamples.buildLeNetGraph(mnistInputName,28,255,mnistTrain.getOutputs().size());
        var mnistImageConverter = new ImageConverter(mnistInputName,28,28,1);


        // Create the CNN trainer
        var mnistCNNTrainer = new TensorFlowTrainer<Label>(mnistCNNTuple.graphDef,
                mnistCNNTuple.outputName, // the name of the logit operation
                gradAlgorithm,            // the gradient descent algorithm
                gradParams,               // the gradient descent hyperparameters
                mnistImageConverter,      // the input feature converter
                mnistOutputConverter,     // the output label converter
                16, // training batch size
                3,  // number of training epochs
                16, // test batch size of the trained model
                -1  // disable logging of the loss value
        );

        // Train the CNN
        var cnnModel = DataProc.train(mnistTrain, mnistCNNTrainer);

        // Evaluate how the neural network did
        var cnnPredictions = cnnModel.predict(mnistTest);
        var cnnEvaluation = labelEval.evaluate(cnnModel,cnnPredictions,mnistTest.getProvenance());
        DataProc.evaluate(mnistTest, cnnModel, (Evaluator) cnnEvaluation);
    }
}
