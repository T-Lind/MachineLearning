package support;

import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.util.Util;

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
}
