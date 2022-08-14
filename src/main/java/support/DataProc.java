package support;

import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.util.Util;

/**
 * Helpful data processing features
 */
public final strictfp class DataProc {
    /**
     * Convert from a DataSource to a MutableDataset.
     * @param source the input datasource (Probably a csvDataSource)
     * @return the MutableDataset version of the same data
     */
    public static Dataset convertSourceSet(DataSource<Label> source){
        return new MutableDataset<>(source);
    }

    public static Model train(MutableDataset dataset, Trainer trainer){
        var linStartTime = System.currentTimeMillis();
        var linModel = trainer.train(dataset);
        var linEndTime = System.currentTimeMillis();
        System.out.println();
        System.out.println(trainer.getClass() + " model training took " + Util.formatDuration(linStartTime,linEndTime));

        return linModel;
    }
}
