package support;

import org.tribuo.*;
import org.tribuo.util.Util;
import org.tribuo.anomaly.*;
import org.tribuo.anomaly.evaluation.*;
import org.tribuo.anomaly.example.GaussianAnomalyDataSource;
import org.tribuo.anomaly.libsvm.*;
import org.tribuo.common.libsvm.*;



public class AnomalyDetection {
    public static void main(String[] args) {
        var eval = new AnomalyEvaluator();

        var data = new MutableDataset<>(new GaussianAnomalyDataSource(2000,/* number of examples */
                0.0f,/*fraction anomalous */
                1L/* RNG seed */));
        var test = new MutableDataset<>(new GaussianAnomalyDataSource(2000,0.2f,2L));

        var params = new SVMParameters<>(new SVMAnomalyType(SVMAnomalyType.SVMMode.ONE_CLASS), KernelType.RBF);
        params.setGamma(1.0);
        params.setNu(0.1);
        var trainer = new LibSVMAnomalyTrainer(params);



        var startTime = System.currentTimeMillis();
        var model = trainer.train(data);
        var endTime = System.currentTimeMillis();
        System.out.println();
        System.out.println("Training took " + Util.formatDuration(startTime,endTime));

        ((LibSVMAnomalyModel)model).getNumberOfSupportVectors();



        var testEvaluation = eval.evaluate(model,test);
        System.out.println(testEvaluation.toString());
        System.out.println(testEvaluation.confusionString());
    }
}
