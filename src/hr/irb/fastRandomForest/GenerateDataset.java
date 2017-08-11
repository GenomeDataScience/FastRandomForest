package hr.irb.fastRandomForest;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.datagenerators.DataGenerator;
import weka.datagenerators.classifiers.classification.BayesNet;
import weka.datagenerators.classifiers.classification.RDG1;
import weka.datagenerators.classifiers.classification.RandomRBF;

import java.io.File;

/**
 * Created by jpique on 08/08/2017.
 */
public class GenerateDataset {
    public static void main(String[] args) throws Exception {
        for (int n_inst = 100; n_inst < 30000; n_inst *= 2) {
            for (int n_feat = 100; n_feat < 30000; n_feat *= 2) {
                if (n_inst * n_feat < 2000*2000) {
//                    String[] options = new String[]{"-o", "C:\\Users\\jpique\\Desktop\\datasets\\dtSet1_" + n_inst + "_" + n_feat + ".arff",
//                    "-S", "" + System.currentTimeMillis()%50, "-n", "" + n_inst, "-a", "" + n_feat, "-c", "2", "-N", ""+ n_feat}; //"-I", "" + 3*n_feat/4,
//                    DataGenerator.runDataGenerator(new RDG1(), options);
                    String[] options = new String[]{"-o", "C:\\Users\\jpique\\Desktop\\datasets\\dtSet3_" + n_inst + "_" + n_feat + ".arff",
                    "-S", "" + System.nanoTime()%50, "-n", "" + n_inst, "-a", "" + n_feat, "-c", "2", "-C", "" + ((int) Utils.log2(n_feat) * 5)};
                    DataGenerator.runDataGenerator(new RandomRBF(), options);
//                    String[] options = new String[]{"-o", "C:\\Users\\jpique\\Desktop\\datasets_aux\\dtSet4_" + n_inst + "_" + n_feat + ".arff",
//                    "-S", "" + System.nanoTime()%50, "-n", "" + n_inst, "-N", "" + n_feat, "-C", "2", "-A", "" + (n_feat*2)};
//                    DataGenerator.runDataGenerator(new BayesNet(), options);
                }
            }
        }
    }
}
