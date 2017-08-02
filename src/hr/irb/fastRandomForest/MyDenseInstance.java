package hr.irb.fastRandomForest;

import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * Created by jpique on 31/07/2017.
 */
public class MyDenseInstance extends DenseInstance {
    public MyDenseInstance(Instance instance) {
        super(instance);
    }

    public MyDenseInstance(double weight, double[] attValues) {
        super(weight, attValues);
    }

    public MyDenseInstance(int numAttributes) {
        super(numAttributes);
    }
}
