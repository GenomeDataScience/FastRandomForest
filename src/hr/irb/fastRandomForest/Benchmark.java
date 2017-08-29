/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Suite 500, Boston, MA 02110.
 */

/*
 *    Benchmark.java
 *    Copyright (C) 2008 Fran Supek
 */
package hr.irb.fastRandomForest;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.experiment.PairedStatsCorrected;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Runs 10 iterations of 10-fold crossvalidation on the supplied arff file(s)
 * using weka.classifiers.trees.RandomForest, and FastRandomForest, and prints
 * accuracy, AUC (averaged over all classes) and computation time to stdout.
 * 
 * As first command-line parameter, supply an arff file, a directory name, or
 * a text file with list of arff datasets.
 * 
 * As second command-line parameter, supply the number of trees in the forests.
 * (-I parameter for the classifiers).
 *
 * As third command-line parameter, supply a comma separated list of number of
 * threads to use, eg. "1,2,4". If ommited, default value is assumed (autodetect
 * number of cores in machine).
 * 
 * @author Fran Supek (fran.supek[AT]irb.hr)
 */
public class Benchmark {

  public static final int numRuns = 5;

  public static final int numFolds = 10;

  private static long myTime = 0;

  private static int numNodes = 0;

  public static synchronized void updateTime (long elapsedTime) {
    myTime += elapsedTime;
  }

  public static synchronized void updateNumNodes (int nNodes) {
    numNodes += nNodes;
  }

  public static void main(String[] args) throws Exception {

    List<File> trainFiles =
            getMatchingFiles(args[0], ".arff");

    List<Integer> threadNums = new ArrayList<Integer>();
    if ( args.length < 3 ) {
      threadNums.add(0);
    } else {
      String[] arr = args[2].split(",");
      for ( String curNum : arr ) {
        threadNums.add( Integer.parseInt(curNum) );
      }
    }

    // create classifiers to compare and set their parameters
    AbstractClassifier[] classifiers = new AbstractClassifier[ threadNums.size() * 2 ];

    for ( int i = 0; i < threadNums.size() * 2; i += 2 ) {

      classifiers[i] = new weka.classifiers.trees.RandomForest();
      classifiers[i].setOptions(new String[]{"-I", "1",
        "-num-slots", Integer.toString(threadNums.get(i)) });
      
      classifiers[i+1] = new hr.irb.fastRandomForest.FastRandomForest();
      classifiers[i+1].setOptions(new String[]{"-I", args[1], // "-import",
        "-threads", Integer.toString(threadNums.get(i))}); // , "-importNew"
      // , "-interactions", "-interactionsNew", "-import", "-importNew"
    }

    StringBuilder s = new StringBuilder("dataset\tnumInstances\tnumNumericAtt\t"
            + "numNominalAtt\tnumClasses");
    for (int i = 0; i < classifiers.length * numRuns; i++) {
      s.append("\tcfr|run\tAUC\tAccy\tmillis");
    }
    s.append("\tsummary");
    System.err.println(s.toString());

    StringBuilder strResultsWeka = new StringBuilder("name, numInstances, numNumeric, numNominal, numClasses, accuracy, time\n");
    StringBuilder strResultsFRF = new StringBuilder("name, numInstances, numNumeric, numNominal, numClasses, accuracy, time\n");

    int idxFile = 1;
    for (File curArff : trainFiles) {
      System.err.println("" + idxFile + "/" + trainFiles.size());
      ++idxFile;

      // load data
      Instances data = new weka.core.converters.ConverterUtils.DataSource(curArff.toString()).getDataSet();
      if (data.classIndex() == -1)
        data.setClassIndex(data.numAttributes() - 1);
      data.deleteWithMissingClass();

      //FastRandomForest frf = new FastRandomForest();
      //frf.setNumTrees(10); frf.setMaxDepth(5);
      //frf.buildClassifier(data);
      
      // count numeric and nominal attributes
      int numNumeric = 0, numNominal = 0;
      for (int i = 0; i < data.numAttributes(); i++) {
        if ( data.classIndex()==i )
          continue;
        if ( data.attribute(i).isNominal() ) 
          numNominal++;
        if ( data.attribute(i).isNumeric() ) 
          numNumeric++;
      }
      System.err.printf( "%s\t%d\t%d\t%d\t%d\t", curArff.getName(), data.numInstances(), numNumeric, numNominal, data.numClasses() );
     

      /* We have adopted a generalization of AUC score to multiclass problems as
       * described in Provost and Domingos (CeDER Working Paper #IS-00-04, Stern
       * School of Business, New York University, 2001), computed by taking a
       * weighted average over all one-vs-all binary classification problems
       * that can be derived from the multiclass problem, where weights
       * correspond to class prior probabilities. */
      double[] classProps = new double[data.numClasses()];
      for ( int i = 0; i < data.numInstances(); i++ )
        classProps[ (int) data.instance(i).classValue() ] += data.instance(i).weight();
      Utils.normalize(classProps);

      double[][] aucScore = new double[classifiers.length][numRuns];
      double[][] accyScore = new double[classifiers.length][numRuns];
      double[][] timeScore = new double[classifiers.length][numRuns];

      for (int curRun = 1; curRun <= numRuns; curRun++) {
        Benchmark.myTime = 0;

        s = new StringBuilder(); 

        for (int curCfr = 0; curCfr < classifiers.length; curCfr++ ) {

          AbstractClassifier aClassifier = classifiers[curCfr];

          Evaluation eval = new Evaluation(data);

          Long millis = System.currentTimeMillis();
          eval.crossValidateModel(aClassifier, data, numFolds, new Random(curRun));
          long elapsedTime = System.currentTimeMillis() - millis;

          double aucSum = 0.0;
          double sumClassProps = 0;
          for (int c = 0; c < data.numClasses(); c++) {
            if (Double.isNaN(eval.areaUnderROC(c)))
              continue;
            aucSum += eval.areaUnderROC(c) * classProps[c];
            // this should sum to 1.0 in the end, as all the classes with AUC==NaN should have weight 0
            sumClassProps += classProps[c];
          }


          aucScore[curCfr][curRun-1] = aucSum / sumClassProps;
          accyScore[curCfr][curRun-1] = eval.pctCorrect();
          timeScore[curCfr][curRun-1] = elapsedTime;

          s.append(String.format( Locale.US, "%02d|%02d\t%.5f\t%.2f\t%6d\t%6d\t",
                  curCfr, curRun, aucSum / sumClassProps,
                  eval.pctCorrect(), elapsedTime, myTime)); // , myTime/(8*1000000)   , numNodes/(500.0 * numFolds)    Utils.log2(numNodes/(500.0 * numFolds))
          numNodes = 0;
          myTime = 0;

          System.gc();

        } // classifier by classifier

        System.err.print(s.toString());
        
      } // run by run

      // write the information (dataset, execTime, accy...) for each file in a StringBuilder
      double sumAccyWeka = 0;
      double sumTimeWeka = 0;
      double sumAccyFRF = 0;
      double sumTimeFRF = 0;
      for (int i = 0; i < numRuns; ++i) {
        sumAccyWeka += accyScore[0][i];
        sumTimeWeka += timeScore[0][i];
        sumAccyFRF += accyScore[1][i];
        sumTimeFRF += timeScore[1][i];
      }
      strResultsWeka.append("" + curArff.getName() + ", " + data.numInstances() + ", " + numNumeric + ", " + numNominal +
              ", " + data.numClasses() + ", " + sumAccyWeka/numRuns + ", " + sumTimeWeka/numRuns + '\n');
      strResultsFRF.append("" + curArff.getName() + ", " + data.numInstances() + ", " + numNumeric + ", " + numNominal +
              ", " + data.numClasses() + ", " + sumAccyFRF/numRuns + ", " + sumTimeFRF/numRuns + '\n');

      // the t-test for accuracy is always performed only between classifier 0 and classifier 1 
      // meaning, the first instance of Weka RF and first instance of FastRF
      // the following instances use a different # of threads but that doesn't affect results
      double testTrainRatio = 1 / (double) (numFolds - 1);
      PairedStatsCorrected pscAuc = new PairedStatsCorrected(0.05, testTrainRatio);
      pscAuc.add(aucScore[0], aucScore[1]);
      pscAuc.calculateDerived();
      PairedStatsCorrected pscAccy = new PairedStatsCorrected(0.05, testTrainRatio);
      pscAccy.add(accyScore[0], accyScore[1]);
      pscAccy.calculateDerived();
      PairedStatsCorrected pscTime = new PairedStatsCorrected(0.05, testTrainRatio);
      pscTime.add(timeScore[0], timeScore[1]);
      pscTime.calculateDerived();

      System.err.printf( Locale.US, "| Statistical significance of difference in mean of " +
              "AUC scores is p=%6.4f (%s wins), " +
              "in accuracy is p=%6.4f (%s wins). " +
              " Average speedup is: %4.2f times.\n",
              pscAuc.differencesProbability, getTextForSignificance( pscAuc.differencesSignificance, "WekaRF", "FastRF" ),
              pscAccy.differencesProbability, getTextForSignificance( pscAccy.differencesSignificance, "WekaRF", "FastRF" ),
              pscTime.xStats.mean / pscTime.yStats.mean
              );
    } // arff by arff

    // write the results (that are in two StringBuilders) in a .csv file
    try{
      PrintWriter writerWeka = new PrintWriter("C:\\Users\\jpique\\Desktop\\results\\resultsWeka_datasetArt48.csv", "UTF-8");
      writerWeka.print(strResultsWeka.toString());
      writerWeka.close();
      PrintWriter writerFRF = new PrintWriter("C:\\Users\\jpique\\Desktop\\results\\resultsFRF_datasetArt66.csv", "UTF-8");
      writerFRF.print(strResultsFRF.toString());
      writerFRF.close();
    } catch (IOException e) {
      // do something
    }

//    Runtime.getRuntime().exec("shutdown -s");
  }

  

  /**
   * When supplied with a directory name, returns an ArrayList with all the
   * files inside that directory that have the specified extension.
   *  
   * When supplied with a single filename...
   * (a) if the extension matches the specified extension, returs an
   *     ArrayList with a single File object inside.
   * (b) if the extension DOES NOT match the specifed extension, treats the
   *     file as a list and extracts filenames from it - one per line - and
   *     returns them within the ArrayList
   */
  public static List<File> getMatchingFiles(
          String fileOrDir, String extension)
          throws FileNotFoundException, IOException {

    ArrayList<File> result = new ArrayList<File>();

    File myFile = new File(fileOrDir);

    String myExt;
    if (extension.length() > 0 && extension.charAt(0) != '.')
      myExt = "." + extension;
    else
      myExt = extension;

    if (!myFile.exists())
      throw new FileNotFoundException(
              "Specified File or directory doesn't exist!");

    if (myFile.isDirectory()) {

      File[] trainFiles;
      trainFiles = new File(fileOrDir).listFiles();
      for (int i = 0; i < trainFiles.length; i++)
        if (trainFiles[i].getName().endsWith(myExt))
          result.add(trainFiles[i]);

    } else if (myFile.getName().endsWith(myExt)) {

      result.add(myFile);

    } else {

      BufferedReader bufRdr = new BufferedReader(new FileReader(fileOrDir));
      String line = null;
      while ((line = bufRdr.readLine()) != null) {
        if (line.endsWith(myExt))
          result.add(new File(line));
        else
          result.add(new File(line + myExt));
      }

    }

    return result;
  }

  
  private static String getTextForSignificance( int significanceFlag, String party1, String party2 ) {
    
    if ( significanceFlag == 0 ) 
      return "noone";
    else if ( significanceFlag > 0 )
      return party1;
    else
      return party2;
    
  }
  

}
