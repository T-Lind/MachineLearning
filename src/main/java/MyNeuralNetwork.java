import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.Dataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.interop.tensorflow.*;
import org.tribuo.interop.tensorflow.example.CNNExamples;
import org.tribuo.interop.tensorflow.example.GraphDefTuple;
import org.tribuo.util.Util;
import support.DataProc;
import support.DataReader;

import java.io.IOException;
import java.util.Map;

public class MyNeuralNetwork {
    public static void main(String[] args) throws IOException {
        var reader = new DataReader(DataProc.DATASETS+"simple-2d-data-train.csv");
        var readerPredict = new DataReader(DataProc.DATASETS+"simple-2d-data-predict.csv");

        Dataset train = reader.generate2dDataDouble("Feature1", "Feature2");
        var test = readerPredict.generate2dDataDouble("Feature1", "Feature2");

        var inputName = "INPUT";

        // Define the parameters of the MLP training algorithm
        var gradAlgorithm = GradientOptimiser.ADAGRAD;
        var gradParams = Map.of("learningRate",0.1f,"initialAccumulatorValue",0.01f);

        var CNNTuple = CNNExamples.buildLeNetGraph(inputName,10,255,train.getOutputs().size());
        var mnistImageConverter = new ImageConverter(inputName,10,10,1);

        var CNNTrainer = new TensorFlowTrainer<Label>(CNNTuple.graphDef,
                CNNTuple.outputName, // the name of the logit operation
                gradAlgorithm,            // the gradient descent algorithm
                gradParams,               // the gradient descent hyperparameters
                mnistImageConverter,      // the input feature converter
                new LabelConverter(),     // the output label converter
                16, // training batch size
                1,  // number of training epochs
                16, // test batch size of the trained model
                -1  // disable logging of the loss value
        );

        // Training the model
        var cnnStart = System.currentTimeMillis();
        var cnnModel = CNNTrainer.train(train);
        var cnnEnd = System.currentTimeMillis();
        System.out.println("MNIST CNN training took " + Util.formatDuration(cnnStart,cnnEnd));

        var labelEval = new LabelEvaluator();

        var cnnPredictions = cnnModel.predict(test);
        var cnnEvaluation = labelEval.evaluate(cnnModel,cnnPredictions,test.getProvenance());
        System.out.println(cnnEvaluation.toString());
        System.out.println(cnnEvaluation.getConfusionMatrix().toString());
    }
}
