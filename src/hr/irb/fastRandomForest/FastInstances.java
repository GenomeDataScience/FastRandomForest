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
    public void takeInstances(int[] listInst, int start, int end) {
        oldInstances = m_Instances;
        temp_m_Instances = new ArrayList<>(end - start + 1);
        for (int i = start; i <= end; ++i) {
            temp_m_Instances.add(m_Instances.get(listInst[i]));
        }
        m_Instances = temp_m_Instances;
    }

    public void takeInstances(boolean[] listInst, int numInBag) {
        oldInstances = m_Instances;
        temp_m_Instances = new ArrayList<>(numInBag);
        for (int i = 0; i < listInst.length; ++i) {
            if (listInst[i]) temp_m_Instances.add(m_Instances.get(i));
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
}
