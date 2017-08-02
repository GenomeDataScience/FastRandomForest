package hr.irb.fastRandomForest;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

/**
 * Created by jpique on 24/07/2017.
 */
public class FastInstances extends Instances {
    protected ArrayList<Instance> oldInstances;
    protected ArrayList<Instance> temp_m_Instances;

    public FastInstances(Reader reader) throws IOException {
        super(reader);
    }

    public FastInstances(Reader reader, int capacity) throws IOException {
        super(reader, capacity);
    }

    public FastInstances(Instances dataset) {
        super(dataset);
    }

    public FastInstances(Instances dataset, int capacity) {
        super(dataset, capacity);
    }

    public FastInstances(Instances source, int first, int toCopy) {
        super(source, first, toCopy);
    }

    public FastInstances(String name, ArrayList<Attribute> attInfo, int capacity) {
        super(name, attInfo, capacity);
    }

    /**
     * Takes the instances indicated by an array of indices. The other original instances are forgotten.
     * @param listInst List of indices that refer to instances of the m_Instances ArrayList
     */
    public void takeInstances(Instance[] listInst, int start, int end) {
        oldInstances = m_Instances;
        temp_m_Instances = new ArrayList<>(end - start + 1);
        for (int i = start; i <= end; ++i) {
            temp_m_Instances.add(listInst[i]);
        }
        m_Instances = temp_m_Instances;
    }

    public void resetInstances() {
        m_Instances = oldInstances;
    }

    public ArrayList<Instance> getInstances() {
        return m_Instances;
    }

    public void setInstances (ArrayList<Instance> instances) {
        m_Instances = instances;
    }

    /** Generates a shallow copy of itself */
    public FastInstances copy() {
        FastInstances newFastInstances = new FastInstances(this, 0);
        newFastInstances.m_Instances = new ArrayList<>(this.m_Instances.size());
        for (Instance inst : this.m_Instances) {
            Instance newInst = (Instance) inst.copy();
            newInst.setWeight(0.0);
            newFastInstances.m_Instances.add(newInst);
        }
        return newFastInstances;
    }

    // S'ha de fer un sort amb un index d'inici i un altre de fi
//    @Override
//    public void sort(int attIndex) {
//        if(!this.attribute(attIndex).isNominal()) {
//            double[] vals = new double[this.numInstances()];
//            Instance[] backup = new Instance[vals.length];
//
//            for(int i = 0; i < vals.length; ++i) {
//                Instance inst = this.instance(i);
//                backup[i] = inst;
//                double val = inst.value(attIndex);
//                if(Utils.isMissingValue(val)) {
//                    vals[i] = 1.7976931348623157E308D;
//                } else {
//                    vals[i] = val;
//                }
//            }
//
//            int[] sortOrder = Utils.sortWithNoMissingValues(vals);
//
//            // TODO Assignar a this.m_Instances un altre ArrayList
//            // S'ha d'evitar modificar l'objecte en si
//            for(int i = 0; i < vals.length; ++i) {
//                this.m_Instances.set(i, backup[sortOrder[i]]);
////                this.temp_m_Instances.set(i, backup[sortOrder[i]]);
//            }
////            this.m_Instances = temp_m_Instances;
//        } else {
//            // TODO Mirar tambe si aquest metode modifica l'objecte
//            this.sortBasedOnNominalAttribute(attIndex);
//        }
//
//    }
}
