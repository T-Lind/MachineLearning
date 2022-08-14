import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.dtree.impurity.*;
import org.tribuo.datasource.*;
import org.tribuo.math.optimisers.*;
import org.tribuo.multilabel.*;
import org.tribuo.multilabel.baseline.*;
import org.tribuo.multilabel.ensemble.*;
import org.tribuo.multilabel.evaluation.*;
import org.tribuo.multilabel.sgd.linear.*;
import org.tribuo.multilabel.sgd.objectives.*;
import org.tribuo.util.Util;
import support.DataProc;
import support.DataReader;

import java.io.IOException;
import java.nio.file.Paths;


public class MultiLabelClassification {
    public static void main(String[] args) throws IOException {
        // Load the training and testing data from a svm file into datasets
        var reader1 = new DataReader("C:\\Users\\zenith\\Documents\\MyDatasets\\yeast_train.svm");
        var reader2 = new DataReader("C:\\Users\\zenith\\Documents\\MyDatasets\\yeast_test.svm");

        var train = reader1.generateSVM();
        var test = reader2.generateSVM();

        // Print data size
        System.out.printf("Training data size = %d, number of features = %d, number of classes = %d%n",train.size(),train.getFeatureMap().size(),train.getOutputInfo().size());
        System.out.printf("Testing data size = %d, number of features = %d, number of classes = %d%n",test.size(),test.getFeatureMap().size(),test.getOutputInfo().size());

        // Create a linear trainer and train it
        var linTrainer = new LinearSGDTrainer(new BinaryCrossEntropy(),new AdaGrad(0.1,0.1),5,1000,1,Trainer.DEFAULT_SEED);
        var linModel = DataProc.train(train, linTrainer);

        // Train the tree model and calculate the time it takes to train
        Trainer<Label> treeTrainer = new CARTClassificationTrainer(6,10,0.0f,1.0f,false,new Entropy(),1L);
        Trainer<MultiLabel> dtTrainer = new IndependentMultiLabelTrainer(treeTrainer);
        var dtModel = DataProc.train(train, dtTrainer);

        // Evaluate the models and calculate the evaluation time
        var eval = new MultiLabelEvaluator();

        DataProc.evaluate(test, linModel, eval);
        DataProc.evaluate(test, dtModel, eval);


    }
}
