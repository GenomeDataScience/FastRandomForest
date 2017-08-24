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
 *    FastRandomTree.java
 *    Copyright (C) 2001 University of Waikato, Hamilton, NZ (original code,
 *      RandomTree.java)
 *    Copyright (C) 2013 Fran Supek (adapted code)
 */

package hr.irb.fastRandomForest;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;


/**
 * Based on the "weka.classifiers.trees.RandomTree" class, revision 1.19,
 * by Eibe Frank and Richard Kirkby, with major modifications made to improve
 * the speed of classifier training.
 * 
 * Please refer to the Javadoc of buildTree, splitData and distribution
 * function, as well as the changelog.txt, for the details of changes to 
 * FastRandomTree.
 * 
 * This class should be used only from within the FastRandomForest classifier.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz) - original code
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz) - original code
 * @author Fran Supek (fran.supek[AT]irb.hr) - adapted code
 * @version $Revision: 0.99$
 */
class FastRandomTree
        extends AbstractClassifier
        implements OptionHandler, WeightedInstancesHandler, Runnable {

  /** for serialization */
  static final long serialVersionUID = 8934314652175299375L;

  public static final double cnst = -999999;

  public boolean[] myInBag;

  public int m_seed;

  public HashSet<Integer> subsetSelectedAttr;
  
  /** The subtrees appended to this tree (node). */
  protected AbstractClassifier[] m_Successors;

  /**
   * For access to parameters of the RF (k, or maxDepth).
   */
  protected FastRandomForest m_MotherForest;

  /** The attribute to split on. */
  protected int m_Attribute = -10000;

  /** The split point. */
  protected double m_SplitPoint = Double.NaN;
  
  /** The proportions of training instances going down each branch. */
  protected double[] m_Prop = null;

  /** Class probabilities from the training vals. */
  protected double[] m_ClassProbs = null;

  /** The dataset used for training. */
  protected transient DataCache data = null;
  
  /**
   * Since 0.99: holds references to temporary arrays re-used by all nodes
   * in the tree, used while calculating the "props" for various attributes in
   * distributionSequentialAtt(). This is meant to avoid frequent 
   * creating/destroying of these arrays.
   */
  protected transient double[] tempProps;  
  
  /**
   * Since 0.99: holds references to temporary arrays re-used by all nodes
   * in the tree, used while calculating the "dists" for various attributes
   * in distributionSequentialAtt(). This is meant to avoid frequent 
   * creating/destroying of these arrays.
   */
  protected transient double[][] tempDists;  
  protected transient double[][] tempDistsOther;
  
  

  /** Minimum number of instances for leaf. */
  protected static final int m_MinNum = 1;

  /**
   * This constructor should not be used. Instead, use the next two constructors
   */
  public FastRandomTree() {
  }

  /**
   * Constructor for the first node of the tree
   * @param motherForest
   * @param data
   */
  public FastRandomTree(FastRandomForest motherForest, DataCache data, int seed) {
    int numClasses = data.numClasses;
    this.m_seed = seed;
    this.data = data;
    // all parameters for training will be looked up in the motherForest (maxDepth, k_Value)
    this.m_MotherForest = motherForest;
    // 0.99: reference to these arrays will get passed down all nodes so the array can be re-used 
    // 0.99: this array is of size two as now all splits are binary - even categorical ones
    this.tempProps = new double[2];
    this.tempDists = new double[2][];
    this.tempDists[0] = new double[numClasses];
    this.tempDists[1] = new double[numClasses];
    this.tempDistsOther = new double[2][];
    this.tempDistsOther[0] = new double[numClasses];
    this.tempDistsOther[1] = new double[numClasses];
  }

  /**
   * Constructor for all the nodes except the root
   * @param motherForest
   * @param data
   * @param tempDists
   * @param tempDistsOther
   * @param tempProps
   */
  public FastRandomTree(FastRandomForest motherForest, DataCache data, double[][] tempDists,
                        double[][] tempDistsOther, double[] tempProps) {
    this.m_MotherForest = motherForest;
    this.data = data;
    // new in 0.99 - used in distributionSequentialAtt()
    this.tempDists = tempDists;
    this.tempDistsOther = tempDistsOther;
    this.tempProps = tempProps;
  }

  /**
   * Get the value of MinNum.
   *
   * @return Value of MinNum.
   */
  public final int getMinNum() {

    return m_MinNum;
  }


  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String KValueTipText() {
    return "Sets the number of randomly chosen attributes.";
  }


  /**
   * Get the value of K.
   * @return Value of K.
   */
  public final int getKValue() {
    return m_MotherForest.m_KValue;
  }


  /**
   * Get the maximum depth of the tree, 0 for unlimited.
   * @return 		the maximum depth.
   */
  public final int getMaxDepth() {
    return m_MotherForest.m_MaxDepth;
  }


  /**
   * Returns default capabilities of the classifier.
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll(); 

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }


  /**
   * This function is not supported by FastRandomTree, as it requires a
   * DataCache for training.

   * @throws Exception every time this function is called
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
    throw new Exception("FastRandomTree can be used only by FastRandomForest " +
            "and FastRfBagger classes, not directly.");
  }



  /**
   * Builds classifier. Makes the initial call to the recursive buildTree 
   * function. The name "run()" is used to support multithreading via an
   * ExecutorService. <p>
   *
   * The "data" field of the FastRandomTree should contain a
   * reference to a DataCache prior to calling this function, and that
   * DataCache should have the "reusableRandomGenerator" field initialized.
   * The FastRfBagging class normally takes care of this before invoking this
   * function.
   */
  public void run() {
    // makes a copy of data and selects randomly which are the inBag instances and the subset of features
    data = data.resample(data.getRandomNumberGenerator(m_seed), m_MotherForest.m_numFeatTree);
    // we need to save the inBag[] array in order to have access to it after this.data is destroyed
    myInBag = data.inBag;

    // compute initial class counts
    double[] classProbs = new double[data.numClasses];
    for (int i = 0; i < data.numInstances; i++) {
      classProbs[data.instClassValues[i]] += data.instWeights[i];
    }

    subsetSelectedAttr = new HashSet<>(data.selectedAttributes.length);
    for (int attr : data.selectedAttributes) subsetSelectedAttr.add(attr);

    // create the attribute indices window - skip class
    int[] attIndicesWindow = data.selectedAttributes;

    // Start with FRF
    if (keepFastRandomTree(data.numInBag)) {
      // We don't need to create the sortedIndices if we won't use it
      data.createInBagSortedIndicesNew();

      buildTree(data.sortedIndices, 0, data.numInBag - 1,
              classProbs, m_Debug, attIndicesWindow, 0);
    }

    // Start with MyRandomTree
    else {
      MyRandomTree auxTree = new MyRandomTree();
      auxTree.setNumFolds(0);
      // Take only the instances that are in bag.
      data.instances.takeInstances(data.inBag, data.numInBag);
      try {
        auxTree.buildTree(data.instances, classProbs, data.selectedAttributes, data.reusableRandomGenerator, getKValue());
      } catch (Exception e) {
        e.printStackTrace();
        System.exit(2);
      }
      m_Successors = new AbstractClassifier[]{auxTree, auxTree};
      m_Attribute = 0;
    }

    this.data = null;
//    int nNodes = countNodes();
//    Benchmark.updateNumNodes(nNodes);
  }

  

  /**
   * Computes class distribution of an instance using the FastRandomTree.<p>
   *
   * In Weka's RandomTree, the distributions were normalized so that all
   * probabilities sum to 1; this would abolish the effect of instance weights
   * on voting. In FastRandomForest 0.97 onwards, the distributions are
   * normalized by dividing with the number of instances going into a leaf.<p>
   * 
   * @param instance the instance to compute the distribution for
   * @return the computed class distribution
   * @throws Exception if computation fails
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    double[] returnedDist = null;

    if (m_Attribute > -1) {  // ============================ node is not a leaf

      if (instance.isMissing(m_Attribute)) {  // ---------------- missing value

        returnedDist = new double[m_MotherForest.m_Info.numClasses()];
        // split instance up
        for (int i = 0; i < m_Successors.length; i++) {
          double[] help = m_Successors[i].distributionForInstance(instance);
          if (help != null) {
            for (int j = 0; j < help.length; j++) {
              returnedDist[j] += m_Prop[i] * help[j];
            }
          }
        }

      } else if (m_MotherForest.m_Info
              .attribute(m_Attribute).isNominal()) { // ------ nominal

        //returnedDist = m_Successors[(int) instance.value(m_Attribute)]
        //        .distributionForInstance(instance);
        
        // 0.99: new - binary splits (also) for nominal attributes
        if ( instance.value(m_Attribute) == m_SplitPoint ) {
          returnedDist = m_Successors[0].distributionForInstance(instance);
        } else {
          returnedDist = m_Successors[1].distributionForInstance(instance);
        }
        
        
      } else { // ------------------------------------------ numeric attributes

        if (instance.value(m_Attribute) < m_SplitPoint) {
          returnedDist = m_Successors[0].distributionForInstance(instance);
        } else {
          returnedDist = m_Successors[1].distributionForInstance(instance);
        }
      }

      return returnedDist;

    } else { // =============================================== node is a leaf

      return m_ClassProbs;

    }

  }


  /**
   * Computes class distribution of an instance using the FastRandomTree. <p>
   *
   * Works correctly only if the DataCache has the same attributes as the one
   * used to train the FastRandomTree - but this function does not check for
   * that! <p>
   * 
   * Main use of this is to compute out-of-bag error (also when finding feature
   * importances).
   * 
   * @param instIdx the index of the instance to compute the distribution for
   * @return the computed class distribution
   * @throws Exception if computation fails
   */
  public double[] distributionForInstanceInDataCache(DataCache data, int instIdx) {
    double[] returnedDist = null;

    if (m_Attribute > -1) {  // ============================ node is not a leaf

      try {
        if ( data.isValueMissing(m_Attribute, instIdx) ) {  // ---------------- missing value

          returnedDist = new double[m_MotherForest.m_Info.numClasses()];
          // split instance up
          for (int i = 0; i < m_Successors.length; i++) {
            double[] help = m_Successors[i] instanceof FastRandomTree ?
                    ((FastRandomTree) m_Successors[i]).distributionForInstanceInDataCache(data, instIdx) :
                    ((MyRandomTree) m_Successors[i]).distributionForInstance(data.instances.get(instIdx));
            if (help != null) {
              for (int j = 0; j < help.length; j++) {
                returnedDist[j] += m_Prop[i] * help[j];
              }
            }
          }

        } else if ( data.isAttrNominal(m_Attribute) ) { // ------ nominal

          //returnedDist = m_Successors[(int) instance.value(m_Attribute)]
          //        .distributionForInstance(instance);

          // 0.99: new - binary splits (also) for nominal attributes
          if ( data.vals[m_Attribute][instIdx] == m_SplitPoint ) {
            returnedDist = m_Successors[0] instanceof FastRandomTree ?
                    ((FastRandomTree) m_Successors[0]).distributionForInstanceInDataCache(data, instIdx) :
                    ((MyRandomTree) m_Successors[0]).distributionForInstance(data.instances.get(instIdx));
          } else {
            returnedDist = m_Successors[1] instanceof FastRandomTree ?
                    ((FastRandomTree) m_Successors[1]).distributionForInstanceInDataCache(data, instIdx) :
                    ((MyRandomTree) m_Successors[1]).distributionForInstance(data.instances.get(instIdx));
          }


        } else { // ------------------------------------------ numeric attributes

          if ( data.vals[m_Attribute][instIdx] < m_SplitPoint) {
            returnedDist = m_Successors[0] instanceof FastRandomTree ?
                    ((FastRandomTree) m_Successors[0]).distributionForInstanceInDataCache(data, instIdx) :
                    ((MyRandomTree) m_Successors[0]).distributionForInstance(data.instances.get(instIdx));
          } else {
            returnedDist = m_Successors[1] instanceof FastRandomTree ?
                    ((FastRandomTree) m_Successors[1]).distributionForInstanceInDataCache(data, instIdx) :
                    ((MyRandomTree) m_Successors[1]).distributionForInstance(data.instances.get(instIdx));
          }
        }

        return returnedDist;

      } catch (Exception e) {
        e.printStackTrace();
        System.exit(2);
        return null;
      }

    } else { // =============================================== node is a leaf
      return m_ClassProbs;
    }
  }
  
  private boolean keepFastRandomTree (int nInstancesNewBranch) {
    if (nInstancesNewBranch == 0) return true;
    else return (getKValue() * Utils.log2(nInstancesNewBranch) / (getKValue() + data.selectedAttributes.length)) > cnst;
  }

  private int countNodes() {
    if (m_Attribute != -1) {
      int result = 1;
      if (m_Successors[0] instanceof FastRandomTree)  {
        result += ((FastRandomTree) m_Successors[0]).countNodes();
      }
      if (m_Successors[1] instanceof FastRandomTree)  {
        result += ((FastRandomTree) m_Successors[1]).countNodes();
      }
      return result;
    } else {
      return 1;
    }
  }
  
 /**
   * Recursively generates a tree. A derivative of the buildTree function from
   * the "weka.classifiers.trees.RandomTree" class, with the following changes
   * made:
   * <ul>
   *
   * <li>m_ClassProbs are now remembered only in leaves, not in every node of
   *     the tree
   *
   * <li>m_Distribution has been removed
   *
   * <li>members of dists, splits, props and vals arrays which are not used are
   *     dereferenced prior to recursion to reduce memory requirements
   *
   * <li>a check for "branch with no training instances" is now (FastRF 0.98)
   *     made before recursion; with the current implementation of splitData(),
   *     empty branches can appear only with nominal attributes with more than
   *     two categories
   *
   * <li>each new 'tree' (i.e. node or leaf) is passed a reference to its
   *     'mother forest', necessary to look up parameters such as maxDepth and K
   *
   * <li>pre-split entropy is not recalculated unnecessarily
   *
   * <li>uses DataCache instead of weka.core.Instances, the reference to the
   *     DataCache is stored as a field in FastRandomTree class and not passed
   *     recursively down new buildTree() calls
   *
   * <li>similarly, a reference to the random number generator is stored
   *     in a field of the DataCache
   *
   * <li>m_ClassProbs are now normalized by dividing with number of instances
   *     in leaf, instead of forcing the sum of class probabilities to 1.0;
   *     this has a large effect when class/instance weights are set by user
   *
   * <li>a little imprecision is allowed in checking whether there was a
   *     decrease in entropy after splitting
   * 
   * <li>0.99: the temporary arrays splits, props, vals now are not wide
   * as the full number of attributes in the dataset (of which only "k" columns
   * of randomly chosen attributes get filled). Now, it's just a single array
   * which gets replaced as the k features are evaluated sequentially, but it
   * gets replaced only if a next feature is better than a previous one.
   * 
   * <li>0.99: the SortedIndices are now not cut up into smaller arrays on every
   * split, but rather re-sorted within the same array in the splitDataNew(),
   * and passed down to buildTree() as the original large matrix, but with
   * start and end points explicitly specified
   * 
   * </ul>
   * 
   * @param sortedIndices the indices of the instances of the whole bootstrap replicate
   * @param startAt First index of the instance to consider in this split; inclusive.
   * @param endAt Last index of the instance to consider; inclusive.
   * @param classProbs the class distribution
   * @param debug whether debugging is on
   * @param attIndicesWindow the attribute window to choose attributes from
   * @param depth the current depth
   */
  protected void buildTree(int[][] sortedIndices, int startAt, int endAt,
          double[] classProbs,
          boolean debug,
          int[] attIndicesWindow,
          int depth)  {

    m_Debug = debug;
    int sortedIndicesLength = endAt - startAt + 1;

    // Check if node doesn't contain enough instances or is pure 
    // or maximum depth reached, make leaf.
    if ( ( sortedIndicesLength < Math.max(2, getMinNum()) )  // small
            || Utils.eq( classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs) )       // pure
            || ( (getMaxDepth() > 0)  &&  (depth >= getMaxDepth()) )                           // deep
            ) {
      m_Attribute = -1;  // indicates leaf (no useful attribute to split on)
      
      // normalize by dividing with the number of instances (as of ver. 0.97)
      // unless leaf is empty - this can happen with splits on nominal
      // attributes with more than two categories
      if ( sortedIndicesLength != 0 )
        for (int c = 0; c < classProbs.length; c++) {
          classProbs[c] /= sortedIndicesLength;
        } 
      m_ClassProbs = classProbs;
      this.data = null;
      return;
    } // (leaf making)
    
    // new 0.99: all the following are for the best attribute only! they're updated while sequentially through the attributes
    double val = Double.NaN; // value of splitting criterion
    double[][] dist = new double[2][data.numClasses];  // class distributions (contingency table), indexed first by branch, then by class
    double[] prop = new double[2]; // the branch sizes (as fraction)
    double split = Double.NaN;  // split point

    // Investigate K random attributes
    int attIndex = 0;
    int windowSize = attIndicesWindow.length;
    int k = getKValue();
    boolean sensibleSplitFound = false;
    double prior = Double.NaN;
    double bestNegPosterior = -Double.MAX_VALUE;
    int bestAttIdx = -1;

    while ((windowSize > 0) && (k-- > 0 || !sensibleSplitFound ) ) {

      int chosenIndex = data.reusableRandomGenerator.nextInt(windowSize);
      attIndex = attIndicesWindow[chosenIndex];

      // shift chosen attIndex out of window
      attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
      attIndicesWindow[windowSize - 1] = attIndex;
      windowSize--;

      // new: 0.99
//      long t = System.nanoTime();
      double candidateSplit = distributionSequentialAtt( prop, dist,
              bestNegPosterior, attIndex, 
              sortedIndices[attIndex], startAt, endAt, classProbs);
//      Benchmark.updateTime(System.nanoTime() - t);


      if ( Double.isNaN(candidateSplit) ) {
        continue;  // we did not improve over a previous attribute! "dist" is unchanged from before
      }
      // by this point we know we have an improvement, so we keep the new split point
      split = candidateSplit;
      bestAttIdx = attIndex;
      
      if ( Double.isNaN(prior) ) { // needs to be computed only once per branch - is same for all attributes (even regardless of missing values)
        prior = SplitCriteria.entropyOverColumns(dist); 
      }
      
      double negPosterior = - SplitCriteria.entropyConditionedOnRows(dist);  // this is an updated dist
      if ( negPosterior > bestNegPosterior ) {  
        bestNegPosterior = negPosterior;
      } else {
        throw new IllegalArgumentException("Very strange!");
      }
      
      val = prior - (-negPosterior); // we want the greatest reduction in entropy
      if ( val > 1e-2 ) {            // we allow some leeway here to compensate
        sensibleSplitFound = true;   // for imprecision in entropy computation
      }
      
    }  // feature by feature in window

    
    if ( sensibleSplitFound ) { 

      m_Attribute = bestAttIdx;   // find best attribute
      m_SplitPoint = split; 
      m_Prop = prop; 
      prop = null; // can be GC'ed

//      long t = System.nanoTime();
      int belowTheSplitStartsAt = splitDataNew(  m_Attribute, m_SplitPoint, sortedIndices, startAt, endAt, dist );
//      Benchmark.updateTime(System.nanoTime() - t);

      m_Successors = new AbstractClassifier[dist.length];  // dist.length now always == 2
      for (int i = 0; i < dist.length; i++) {
        // number of instances of the successor that will be created
        int nInstSucc = i == 0 ? belowTheSplitStartsAt - startAt : endAt - belowTheSplitStartsAt + 1;

        // continue with the FastRandomTree if --> (K * log(nInst)) / (K + nFeat) > some constant value
        if (keepFastRandomTree(nInstSucc)) {
          FastRandomTree auxTree = new FastRandomTree(m_MotherForest, data, tempDists, tempDistsOther, tempProps);

          // check if we're about to make an empty branch - this can happen with
          // nominal attributes with more than two categories (as of ver. 0.98)
          if (belowTheSplitStartsAt - startAt == 0) {
            // in this case, modify the chosenAttDists[i] so that it contains
            // the current, before-split class probabilities, properly normalized
            // by the number of instances (as we won't be able to normalize
            // after the split)
            for (int j = 0; j < dist[i].length; j++)
              dist[i][j] = classProbs[j] / sortedIndicesLength;
          }

          if (i == 0) {   // before split
            auxTree.buildTree(sortedIndices, startAt, belowTheSplitStartsAt - 1,
                    dist[i], m_Debug, attIndicesWindow, depth + 1);
          } else {  // after split
            auxTree.buildTree(sortedIndices, belowTheSplitStartsAt, endAt,
                    dist[i], m_Debug, attIndicesWindow, depth + 1);
          }

          dist[i] = null;
          m_Successors[i] = auxTree;
        }
        // Change to MyRandomTree
        else {
          MyRandomTree auxTree = new MyRandomTree();
          auxTree.setNumFolds(0);
          // Take only the instances that belong to this node. Drop the others.
          if (i == 0) {
            data.instances.takeInstances(sortedIndices[m_Attribute], startAt, belowTheSplitStartsAt - 1);
          } else {
            data.instances.takeInstances(sortedIndices[m_Attribute], belowTheSplitStartsAt, endAt);
          }
          try {
            auxTree.buildTree(data.instances, dist[i], data.selectedAttributes, data.reusableRandomGenerator, getKValue());
          } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
          }
          m_Successors[i] = auxTree;
          // restore all the instances, because the data object is reused for the other branches of this tree
          data.instances.resetInstances();
        }
      }
      sortedIndices = null;
      
    } else { // ------ make leaf --------

      m_Attribute = -1;
      
      // normalize by dividing with the number of instances (as of ver. 0.97)
      // unless leaf is empty - this can happen with splits on nominal attributes
      if ( sortedIndicesLength != 0 )
        for (int c = 0; c < classProbs.length; c++) {
          classProbs[c] /= sortedIndicesLength;
        }
      m_ClassProbs = classProbs;
    }
    this.data = null; // dereference all pointers so data can be GC'd after tree is built
  }



//  /**
//   * Computes size of the tree.
//   *
//   * @return the number of nodes
//   */
//  public int numNodes() {
//
//    if (m_Attribute == -1) {
//      return 1;
//    } else {
//      int size = 1;
//      for (int i = 0; i < m_Successors.length; i++) {
//        size += m_Successors[i].numNodes();
//      }
//      return size;
//    }
//  }


  
  /**
   * Splits instances into subsets; new for FastRF 0.99. Does not create new
   * arrays with split indices, but rather reorganizes the indices within the 
   * supplied sortedIndices to conform with the split. Works only within given
   * boundaries. <p>
   * 
   * Note: as of 0.99, all splits (incl. categorical) are always binary.
   *
   * @param att the attribute index
   * @param splitPoint the splitpoint for numeric attributes
   * @param sortedIndices the sorted indices of the whole set - gets overwritten!
   * @param startAt Inclusive, 0-based index. Does not touch anything before this value.
   * @param endAt  Inclusive, 0-based index. Does not touch anything after this value.
   * @param dist  dist[0] -> will have the counts of instances for the first branch.
   *              dist[1] -> will have the counts of instances for the second branch.
   *
   * @return the first index of the "below the split" instances
   */
  protected int splitDataNew(
          int att, double splitPoint,
          int[][] sortedIndices, int startAt, int endAt, double[][] dist ) {

    Random random = data.reusableRandomGenerator;
    int j;
    // 0.99: we have binary splits also for nominal data
    int[] num = new int[2]; // how many instances go to each branch
    // we might possibly want to recycle this array for the whole tree
    int[] tempArr = new int[ endAt-startAt+1 ];
    Arrays.fill(dist[0], 0); Arrays.fill(dist[1], 0);

    if ( data.isAttrNominal(att) ) { // ============================ if nominal
      int auxAtt = data.attInSortedIndices[0];
      for (j = startAt; j <= endAt; j++) {
        int inst = sortedIndices[auxAtt][j];
        int branch;
        if ( data.isValueMissing(att, inst) ) { // ---------- has missing value
          // decide where to put this instance randomly, with bigger branches getting a higher chance
          branch = ( random.nextDouble() > m_Prop[0] ) ? 1 : 0;
        } else { // ----------------------------- does not have missing value
          // if it matches the category to "split out", put above split all other categories go below split
          branch = ( data.vals[att][inst] == splitPoint ) ? 0 : 1;
        } // --------------------------------------- end if has missing value
        data.whatGoesWhere[ inst ] = branch;
        // compute the correct value for dist when we know where the instances with missing values go
        // the value calculated in distrib...Att() is not exact, so we have to calculate the correct one
        dist[branch][data.instClassValues[inst]] += data.instWeights[inst];
        num[branch] += 1;
      }

    } else { // =================================================== if numeric
      for (j = startAt; j <= endAt ; j++) {
        int inst = sortedIndices[att][j];
        int branch;
        if ( data.isValueMissing(att, inst) ) { // ---------- has missing value
          // decide where to put this instance randomly, with bigger branches getting a higher chance
          double rn = random.nextDouble();
          branch = ( rn > m_Prop[0] ) ? 1 : 0;
        } else { // ----------------------------- does not have missing value
          branch = ( data.vals[att][inst] < splitPoint ) ? 0 : 1;
        } // --------------------------------------- end if has missing value
        data.whatGoesWhere[ inst ] = branch;
        dist[branch][data.instClassValues[inst]] += data.instWeights[inst];
        num[branch] += 1;
      } // end for instance by instance
    }  // ============================================ end if nominal / numeric

//    // TODO Test if this works
//    boolean keepTreeBranch0 = keepFastRandomTree(num[0]);
//    boolean keepTreeBranch1 = keepFastRandomTree(num[1]);
//    int[] selectedAttributes;
//
//    if (keepTreeBranch0 || keepTreeBranch1) selectedAttributes = data.selectedAttributes;
//      // we don't need to rebuild the sortedIndices[][] matrix if we will change both branches to MyRandomTree
//    else selectedAttributes = new int[]{att};

    for (int a : data.attInSortedIndices) { // xxxxxxxxxx attr by attr

      // the first index of the sortedIndices in the above branch, and the first index in the below
      int startAbove = startAt, startBelow = 0; // always only 2 sub-branches, remember where second starts

      // fill them with stuff by looking at goesWhere array
      for (j = startAt; j <= endAt; j++) {

        int inst = sortedIndices[ a ][j];
        int branch = data.whatGoesWhere[ inst ];  // can be only 0 or 1

        if ( branch==0 ) {
          sortedIndices[a][startAbove] = sortedIndices[a][j];
          startAbove++;
        } else {
          tempArr[startBelow] = sortedIndices[a][j];
          startBelow++;
        }
      }

      // now copy the tempArr into the sortedIndices, thus overwriting it
      System.arraycopy( tempArr, 0, sortedIndices[a], startAt+num[0], num[1] );

    } // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx end for attr by attr

    return startAt+num[0]; // the first index of "below the split" instances
  }


  /**
   * Computes class distribution for an attribute. New in FastRF 0.99, main
   * changes:
   * <ul>
   *   <li> now reuses the temporary counting arrays (this.tempDists,
   *   this.tempDistsOthers) instead of creating/destroying arrays
   *   <li> does not create a new "dists" for each attribute it examines; instead
   *   it replaces the existing "dists" (supplied as a parameter) but only if the
   *   split is better than the previous best split
   *   <li> always creates binary splits, even for categorical variables; thus
   *   might give slightly different classification results than the old
   *   RandomForest
   * </ul>
   *
   * @param propsBestAtt gets filled with relative sizes of branches (total = 1)
   * for the best examined attribute so far; updated ONLY if current attribute is
   * better that the previous best
   * @param distsBestAtt these are the contingency matrices for the best examined
   * attribute so far; updated ONLY if current attribute is better that the previous best
   * @param scoreBestAtt Checked against the score of the attToExamine to determine
   * if the propsBestAtt and distsBestAtt need to be updated.
   * @param attToExamine the attribute index (which one to examine, and change the above
   * matrices if the attribute is better than the previous one)
   * @param sortedIndicesOfAtt the sorted indices of the vals for the attToExamine.
   * @param startAt Index in sortedIndicesOfAtt; do not touch anything below this index.
   * @param endAt Index in sortedIndicesOfAtt; do not touch anything after this index.
   */
  protected double distributionSequentialAtt( double[] propsBestAtt, double[][] distsBestAtt,
                                              double scoreBestAtt, int attToExamine, int[] sortedIndicesOfAtt,
                                              int startAt, int endAt, double[] classProbs ) {

    double splitPoint = -Double.MAX_VALUE;

    // a contingency table of the split point vs class.
    double[][] dist = this.tempDists;
    double[][] currDist = this.tempDistsOther;
    // Copy the current class distribution
    for (int i = 0; i < classProbs.length; ++i) {
      currDist[1][i] = classProbs[i];
    }

    double[] props = this.tempProps;

    int i;
    int sortedIndicesOfAttLength = endAt - startAt + 1;

    if ( data.isAttrNominal(attToExamine) ) { // ====================== nominal attributes

      // 0.99: new routine - makes a one-vs-all split on categorical attributes

      int numLvls = data.attNumVals[attToExamine];
      int bestLvl = 0; // the index of the category which is best to "split out"
      int idxMissVal = 0;
      sortedIndicesOfAtt = data.sortedIndices[data.attInSortedIndices[0]];

      // note: if we have only two levels, it doesn't matter which one we "split out"
      // we can thus safely check only the first one
      if ( numLvls <= 2 ) {
        Arrays.fill( dist[0], 0.0 ); Arrays.fill( dist[1], 0.0 );
        bestLvl = 0; // this means that the category with index 0 always
        // goes 'above' the split and category with index 1 goes 'below' the split
        for (i = startAt; i <= endAt; i++) {
          int inst = sortedIndicesOfAtt[i];
          if (! data.isValueMissing(attToExamine, inst)) {
            dist[ (int)data.vals[attToExamine][inst] ][ data.instClassValues[inst] ] += data.instWeights[inst];
          } else {
            data.instancesMissVal[idxMissVal] = inst; ++idxMissVal;
          }
        }
        if (idxMissVal == sortedIndicesOfAttLength) return Double.NaN; // all values missing

      } else {   // for >2 levels, we have to search different splits
        // fill a matrix that has the number of instances of lvl "i" and class "j"
        double[][] levelsClasses = data.levelsClasses;
        for (i = 0; i < numLvls; ++i) Arrays.fill(levelsClasses[i], 0.0);

        for (i = startAt; i <= endAt; i++) {
          int inst = sortedIndicesOfAtt[i];
          if (! data.isValueMissing(attToExamine, inst)) {
            levelsClasses[(int) data.vals[attToExamine][inst]] [data.instClassValues[inst]] += data.instWeights[inst];
          } else { // we work now only with non missing values
            currDist[1][data.instClassValues[inst]] -= data.instWeights[inst];
            data.instancesMissVal[idxMissVal] = inst; ++idxMissVal;
          }
        }
        if (idxMissVal == sortedIndicesOfAttLength) return Double.NaN; // all values missing
        // entropy..Rows(levelsClasses, numLvls)
        // com decideixo el splitPoint? Hauria de ser un array. O podria ser a numLvls/2.

        // copy the values of currDist[1] to dist[1]
        System.arraycopy(currDist[1], 0, dist[1], 0, currDist[1].length);

        // TODO Here we should implement the total split, not the one vs all
        currDist[0] = levelsClasses[0];
        for (i = 0; i < data.numClasses; ++i) {
          currDist[1][i] -= levelsClasses[0][i];
        }
        double currVal = -SplitCriteria.entropyConditionedOnRows(currDist);; // current value of splitting criterion
        double bestVal = currVal; // best value of splitting criterion

        for ( int lvl = 1; lvl < numLvls; lvl++ ) {

          currDist[0] = levelsClasses[lvl];
          for (i = 0; i < data.numClasses; ++i) {
            currDist[1][i] += levelsClasses[lvl-1][i];
            currDist[1][i] -= levelsClasses[lvl][i];
          }

          // we filled the "dist" for the current level, find score and see if we like it
          currVal = -SplitCriteria.entropyConditionedOnRows(currDist);
          if ( currVal > bestVal ) {
            bestVal = currVal;
            bestLvl = lvl;
          }
        }  // examine how well "splitting out" of individual levels works for us

        // remember the contingency table from the best "lvl" and store it in "dist"
        for (i = 0; i < data.numClasses; ++i) {
          dist[0][i] = levelsClasses[bestLvl][i];
          dist[1][i] -= levelsClasses[bestLvl][i];
        }
      }

      splitPoint = bestLvl; // signals we've found a sensible split point; by
      // definition, a split on a nominal attribute will always be sensible

      // compute total weights for each branch (= props)
      // again, we reuse the tempProps of the tree not to create/destroy new arrays
      countsToFreqs(dist, props);  // props gets overwritten, previous contents don't matters
      // distribute *counts* of instances with missing values using the "props"
      for (i = 0; i < idxMissVal; ++i) {//for (i = 0; i < idxMissVal; ++i) {
        int inst = data.instancesMissVal[i];
        dist[ 0 ][ data.instClassValues[inst] ] += props[ 0 ] * data.instWeights[ inst ] ;
        dist[ 1 ][ data.instClassValues[inst] ] += props[ 1 ] * data.instWeights[ inst ] ;
      }

    } else { // ============================================ numeric attributes
      Arrays.fill( currDist[0], 0.0 );

      // find how many missing values we have for this attribute (they're always at the end)
      // update the distribution to the future second son
      int lastNonmissingValIdx = endAt;
      for (int j = endAt; j >= startAt; j-- ) {
        int inst = sortedIndicesOfAtt[j];
        if ( data.isValueMissing(attToExamine, sortedIndicesOfAtt[j]) ) {
          currDist[1][data.instClassValues[inst]] -= data.instWeights[inst];
          lastNonmissingValIdx = j-1;
        } else {
          break;
        }
      }
      if ( lastNonmissingValIdx < startAt ) {  // only missing values in this feature??
        return Double.NaN; // we cannot split on it
      }

      copyDists(currDist, dist);

      double currVal = -Double.MAX_VALUE; // current value of splitting criterion
      double bestVal = -Double.MAX_VALUE; // best value of splitting criterion
      int bestI = 0; // the value of "i" BEFORE which the splitpoint is placed

      for (i = startAt+1; i <= lastNonmissingValIdx; i++) {  // --- try all split points

        int inst = sortedIndicesOfAtt[i];
        int prevInst = sortedIndicesOfAtt[i-1];

        currDist[0][ data.instClassValues[ prevInst ] ] += data.instWeights[ prevInst ] ;
        currDist[1][ data.instClassValues[ prevInst ] ] -= data.instWeights[ prevInst ] ;

        // do not allow splitting between two instances with the same class or with the same value
        if (data.instClassValues[prevInst] != data.instClassValues[inst] && data.vals[attToExamine][inst] > data.vals[attToExamine][prevInst] ) {
          currVal = -SplitCriteria.entropyConditionedOnRows(currDist);
          if (currVal > bestVal) {
            bestVal = currVal;
            bestI = i;
          }
        }
      }                                             // ------- end trying split points

      /*
       * Determine the best split point:
       * bestI == 0 only if all instances had missing values, or there were
       * less than 2 instances; splitPoint will remain set as -Double.MAX_VALUE.
       * This is not really a useful split, as all of the instances are 'below'
       * the split line, but at least it's formally correct. And the dists[]
       * also has a default value set previously.
       */
      if ( bestI > startAt ) { // ...at least one valid splitpoint was found

        int instJustBeforeSplit = sortedIndicesOfAtt[bestI-1];
        int instJustAfterSplit = sortedIndicesOfAtt[bestI];
        splitPoint = ( data.vals[ attToExamine ][ instJustAfterSplit ]
                + data.vals[ attToExamine ][ instJustBeforeSplit ] ) / 2.0;

        // now make the correct dist[] (for the best split point) from the
        // default dist[] (all instances in the second branch, by iterating
        // through instances until we reach bestI, and then stop.
        for ( int ii = startAt; ii < bestI; ii++ ) {
          int inst = sortedIndicesOfAtt[ii];
          dist[0][ data.instClassValues[ inst ] ] += data.instWeights[ inst ] ;
          dist[1][ data.instClassValues[ inst ] ] -= data.instWeights[ inst ] ;
        }
      }

      // compute total weights for each branch (= props)
      // again, we reuse the tempProps of the tree not to create/destroy new arrays
      countsToFreqs(dist, props);  // props gets overwritten, previous contents don't matters
      // distribute *counts* of instances with missing values using the "props"
      // start 1 after the non-missing val (if there is anything)
      for (i = lastNonmissingValIdx + 1; i <= endAt; ++i) {
        int inst = sortedIndicesOfAtt[i];
        dist[ 0 ][ data.instClassValues[inst] ] += props[ 0 ] * data.instWeights[ inst ] ;
        dist[ 1 ][ data.instClassValues[inst] ] += props[ 1 ] * data.instWeights[ inst ] ;
      }
    } // ================================================== nominal or numeric?

    // update the distribution after split and best split point
    // but ONLY if better than the previous one -- we need to recalculate the
    // entropy (because this changes after redistributing the instances with
    // missing values in the current attribute). Also, for categorical variables
    // it was not calculated before.
    double curScore = -SplitCriteria.entropyConditionedOnRows(dist);
    if ( curScore > scoreBestAtt && splitPoint > -Double.MAX_VALUE ) {  // overwrite the "distsBestAtt" and "propsBestAtt" with current values
      copyDists(dist, distsBestAtt);
      System.arraycopy( props, 0, propsBestAtt, 0, props.length );
      return splitPoint;
    } else {
      // returns a NaN instead of the splitpoint if the attribute was not better than a previous one.
      return Double.NaN;
    }
  }

  /**
   * Normalizes branch sizes so they contain frequencies (stored in "props")
   * instead of counts (stored in "dist"). Creates a new double[] which it 
   * returns.
   */  
  protected static double[] countsToFreqs( double[][] dist ) {
    
    double[] props = new double[dist.length];
    
    for (int k = 0; k < props.length; k++) {
      props[k] = Utils.sum(dist[k]);
    }
    if (Utils.eq(Utils.sum(props), 0)) {
      for (int k = 0; k < props.length; k++) {
        props[k] = 1.0 / (double) props.length;
      }
    } else {
      FastRfUtils.normalize(props);
    }
    return props;
  }

  /**
   * Normalizes branch sizes so they contain frequencies (stored in "props")
   * instead of counts (stored in "dist"). <p>
   * 
   * Overwrites the supplied "props"! <p>
   * 
   * props.length must be == dist.length.
   */  
  protected static void countsToFreqs( double[][] dist, double[] props ) {
    
    for (int k = 0; k < props.length; k++) {
      props[k] = Utils.sum(dist[k]);
    }
    if (Utils.eq(Utils.sum(props), 0)) {
      for (int k = 0; k < props.length; k++) {
        props[k] = 1.0 / (double) props.length;
      }
    } else {
      FastRfUtils.normalize(props);
    }

  }

  
  /**
   * Makes a copy of a "dists" array, which is a 2 x numClasses array. 
   * 
   * @param distFrom
   * @param distTo Gets overwritten.
   */
  protected static void copyDists( double[][] distFrom, double[][] distTo ) {
    for ( int i = 0; i < distFrom[0].length; i++ ) {
      distTo[0][i] = distFrom[0][i];
    }
    for ( int i = 0; i < distFrom[1].length; i++ ) {
      distTo[1][i] = distFrom[1][i];
    }
  }

  
  
  /**
   * Main method for this class.
   * 
   * @param argv the commandline parameters
   */
  public static void main(String[] argv) {
    runClassifier(new FastRandomTree(), argv);
  }



  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 0.99$");
  }


  
}

