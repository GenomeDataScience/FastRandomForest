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
 *    FastRandomForest.java
 *    Copyright (C) 2001 University of Waikato, Hamilton, NZ (original code,
 *      RandomForest.java )
 *    Copyright (C) 2009 Fran Supek (adapted code)
 */

package hr.irb.fastRandomForest;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.ExecutionException;

/**
 * Based on the "weka.classifiers.trees.RandomForest" class, revision 1.12,
 * by Richard Kirkby, with minor modifications:
 * <p/>
 * - uses FastRfBagger with FastRandomTree, instead of Bagger with RandomTree.
 * - stores dataset header (instead of every Tree storing its own header)
 * - checks if only ZeroR model is possible (instead of each Tree checking)
 * - added "-threads" option
 * <p/>
 * <!-- globalinfo-start -->
 * Class for constructing a forest of random trees.<br/>
 * <br/>
 * For more information see: <br/>
 * <br/>
 * Leo Breiman (2001). Random Forests. Machine Learning. 45(1):5-32.
 * <p/>
 * <!-- globalinfo-end -->
 * <p/>
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman2001,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {5-32},
 *    title = {Random Forests},
 *    volume = {45},
 *    year = {2001}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * <p/>
 * <!-- options-start -->
 * Valid options are: <p/>
 * <p/>
 * <pre> -I &lt;number of trees&gt;
 *  Number of trees to build.</pre>
 * <p/>
 * <pre> -K &lt;number of features&gt;
 *  Number of features to consider (&lt;1=int(logM+1)).</pre>
 * <p/>
 * <pre> -S
 *  Seed for random number generator.
 *  (default 1)</pre>
 * <p/>
 * <pre> -depth &lt;num&gt;
 *  The maximum depth of the trees, 0 for unlimited.
 *  (default 0)</pre>
 * <p/>
 * <pre> -numFeatTree &lt;num&gt;
 *  Number of features selected for each tree.</pre>
 * <p/>
 * <pre> -import
 *  Compute and output RF feature importances (slow).</pre>
 * <p/>
 * <pre> -importNew
 *  Compute and output RF dropout importance (slow).</pre>
 * <p/>
 * <pre> -interactions
 *  Compute and output RF interactions (very slow).</pre>
 * <p/>
 * <pre> -interactionsNew
 *  Compute and output RF interactions using the new version (very slow).</pre>
 * <p/>
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * <p/>
 * <!-- options-end -->
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz) - original code
 * @author Fran Supek (fran.supek[AT]irb.hr) - adapted code
 * @author Jordi Pique (2.0 version)
 * @version $Revision: 2.0$
 */
public class FastRandomForest
  extends AbstractClassifier
  implements OptionHandler, Randomizable, WeightedInstancesHandler,
             AdditionalMeasureProducer, TechnicalInformationHandler{

  /** for serialization */
  static final long serialVersionUID = 4216839470751428700L;

  /** Number of trees in forest. */
  protected int m_numTrees = 100;

  /** Minimum expected number of trees with a given features */
  protected int minTrees = 20;

  /** Number of attributes in the dataset */
  protected int m_numAttributes;

  /** Number of features selected to construct a tree */
  protected int m_numFeatTree = 0;

  /** The random seed. */
  protected int m_randomSeed = 1;

  /** Number of features to consider in random feature selection. If less than 1 will use int(logM+1) ) */
  protected int m_KValue = 0;

  /** Number of simultaneous threads to use in computation (0 = autodetect). */
  protected int m_NumThreads = 0;

  /** The bagger. */
  protected FastRfBagging m_bagger = null;

  /** The maximum depth of the trees (0 = unlimited) */
  protected int m_MaxDepth = 0;

  /** The header information. */
  protected Instances m_Info = null;

  /** a ZeroR model in case no model can be built from the data */
  protected AbstractClassifier m_ZeroR;

  /**
   * Returns a string describing classifier
   *
   * @return a description suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String globalInfo(){

    return
      "Class for constructing a forest of random trees.\n\n"
        + "For more information see: \n\n"
        + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   *
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation(){
    TechnicalInformation result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "2001");
    result.setValue(Field.TITLE, "Random Forests");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "45");
    result.setValue(Field.NUMBER, "1");
    result.setValue(Field.PAGES, "5-32");

    return result;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String numTreesTipText(){
    return "The number of trees to be generated.";
  }

  /**
   * Get the value of numTrees.
   *
   * @return Value of numTrees.
   */
  public int getNumTrees(){

    return m_numTrees;
  }

  /**
   * Set the value of numTrees.
   *
   * @param newNumTrees Value to assign to numTrees.
   */
  public void setNumTrees(int newNumTrees){

    m_numTrees = newNumTrees;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String numFeaturesTipText(){
    return "The number of attributes to be used in random selection (see RandomTree2).";
  }

  /**
   * Get the number of features used in random selection.
   *
   * @return Value of numFeatures.
   */
  public int getNumFeatures(){

    return m_KValue;
  }

  /**
   * Set the number of features to use in random selection.
   *
   * @param newNumFeatures Value to assign to numFeatures.
   */
  public void setNumFeatures(int newNumFeatures){

    m_KValue = newNumFeatures;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String seedTipText(){
    return "The random number seed to be used.";
  }

  /**
   * Set the seed for random number generation.
   *
   * @param seed the seed
   */
  public void setSeed(int seed){

    m_randomSeed = seed;
  }

  /**
   * Gets the seed for the random number generations
   *
   * @return the seed for the random number generation
   */
  public int getSeed(){

    return m_randomSeed;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String maxDepthTipText(){
    return "The maximum depth of the trees, 0 for unlimited.";
  }

  /**
   * Get the maximum depth of trh tree, 0 for unlimited.
   *
   * @return the maximum depth.
   */
  public int getMaxDepth(){
    return m_MaxDepth;
  }

  /**
   * Set the maximum depth of the tree, 0 for unlimited.
   *
   * @param value the maximum depth.
   */
  public void setMaxDepth(int value){
    m_MaxDepth = value;
  }


  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String numThreadsTipText(){
    return "Number of simultaneous threads to use in computation (0 = autodetect).";
  }

  /**
   * Get the number of simultaneous threads used in training, 0 for autodetect.
   *
   * @return the maximum depth.
   */
  public int getNumThreads(){
    return m_NumThreads;
  }

  /**
   * Set the number of simultaneous threads used in training, 0 for autodetect.
   *
   * @param value the maximum depth.
   */
  public void setNumThreads(int value){
    m_NumThreads = value;
  }

  ////////////////////////////
  // Feature importances stuff
  ////////////////////////////

  /**
   * The value of the features importances.
   */
  private double[] m_FeatureImportances;
  
  /**
   * Whether to compute the importances or not.
   */
  private boolean m_computeImportances = false;

  /**
   * Whether to compute the importances or not using dropout importance.
   */
  private boolean m_computeDropoutImportance = false;

  /**
   * Whether to compute the interactions or not.
   */
  private boolean m_computeInteractions = false;

  /**
   * Whether to compute the interactions or not using the new method.
   */
  private boolean m_computeInteractionsNew = false;



  

  /**
   * Gets the out of bag error that was calculated as the classifier was built.
   *
   * @return the out of bag error
   */
  public double measureOutOfBagError(){

    if(m_bagger != null){
      return m_bagger.measureOutOfBagError();
    }
    else return Double.NaN;
  }

  /**
   * Returns an enumeration of the additional measure names.
   *
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures(){

    Vector newVector = new Vector(1);
    newVector.addElement("measureOutOfBagError");
    return newVector.elements();
  }

  /**
   * Returns the value of the named measure.
   *
   * @param additionalMeasureName the name of the measure to query for its value
   *
   * @return the value of the named measure
   *
   * @throws IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName){

    if(additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")){
      return measureOutOfBagError();
    }
    else{
      throw new IllegalArgumentException(additionalMeasureName
        + " not supported (FastRandomForest)");
    }
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions(){

    Vector newVector = new Vector();

    newVector.addElement(new Option(
      "\tNumber of trees to build.",
      "I", 1, "-I <number of trees>"));

    newVector.addElement(new Option(
      "\tNumber of features to consider (<1=int(logM+1)).",
      "K", 1, "-K <number of features>"));

    newVector.addElement(new Option(
      "\tSeed for random number generator.\n"
        + "\t(default 1)",
      "S", 1, "-S"));

    newVector.addElement(new Option(
      "\tThe maximum depth of the trees, 0 for unlimited.\n"
        + "\t(default 0)",
      "depth", 1, "-depth <num>"));

    newVector.addElement(new Option(
      "\tThe number of simultaneous threads to use for computation, 0 for autodetect.\n"
        + "\t(default 0)",
      "threads", 1, "-threads <num>"));

    newVector.addElement(new Option(
      "\tWhether to compute feature importances.\n",
      "import", 0, "-import"));
    
    Enumeration enu = super.listOptions();
    while(enu.hasMoreElements()){
      newVector.addElement(enu.nextElement());
    }

    return newVector.elements();
  }

  /**
   * Gets the current settings of the forest.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions(){
    Vector result;
    String[] options;
    int i;

    result = new Vector();

    result.add("-I");
    result.add("" + getNumTrees());

    result.add("-K");
    result.add("" + getNumFeatures());

    result.add("-S");
    result.add("" + getSeed());

    if(getMaxDepth() > 0){
      result.add("-depth");
      result.add("" + getMaxDepth());
    }

    if(getNumThreads() > 0){
      result.add("-threads");
      result.add("" + getNumThreads());
    }
    
    if (getComputeImportances()) {
      result.add("-import");
    }    

    options = super.getOptions();
    for(i = 0; i < options.length; i++)
      result.add(options[i]);

    return (String[])result.toArray(new String[result.size()]);
  }


  /**
   * Parses a given list of options. <p/>
   * <p/>
   * <!-- options-start -->
   * Valid options are: <p/>
   * <p/>
   * <pre> -I &lt;number of trees&gt;
   *  Number of trees to build.</pre>
   * <p/>
   * <pre> -K &lt;number of features&gt;
   *  Number of features to consider in a node(&lt;1=int(logM+1)).</pre>
   * <p/>
   * <pre> -S
   *  Seed for random number generator.
   *  (default 1)</pre>
   * <p/>
   * <pre> -depth &lt;num&gt;
   *  The maximum depth of the trees, 0 for unlimited.
   *  (default 0)</pre>
   * <p/>
   * <pre> -threads
   *  Number of simultaneous threads to use.
   *  (default 0 = autodetect number of available cores)</pre>
   * <p/>
   * <pre> -numFeatTree
   *  Number of features selected for each tree.</pre>
   * <p/>
   * <pre> -import
   *  Compute and output RF feature importances (slow).</pre>
   * <p/>
   * <pre> -importNew
   *  Compute and output RF feature importances using the new version (slow).</pre>
   * <p/>
   * <pre> -interactions
   *  Compute and output RF interactions (very slow).</pre>
   * <p/>
   * <pre> -interactionsNew
   *  Compute and output RF interactions using the new version (very slow).</pre>
   * <p/>
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * <p/>
   * <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   *
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception{
    String tmpStr;

    tmpStr = Utils.getOption('I', options);
    if ( tmpStr.length() != 0 ) {
      m_numTrees = Integer.parseInt(tmpStr);
    } else {
      m_numTrees = 10;
    }

    tmpStr = Utils.getOption('K', options);
    if ( tmpStr.length() != 0 ) {
      m_KValue = Integer.parseInt(tmpStr);
    } else {
      m_KValue = 0;
    }

    tmpStr = Utils.getOption('S', options);
    if ( tmpStr.length() != 0 ) {
      setSeed(Integer.parseInt(tmpStr));
    }
    else{
      setSeed(1);
    }

    tmpStr = Utils.getOption("depth", options);
    if ( tmpStr.length() != 0 ){
      setMaxDepth(Integer.parseInt(tmpStr));
    } else {
      setMaxDepth(0);
    }

    tmpStr = Utils.getOption("threads", options);
    if ( tmpStr.length() != 0 ){
      setNumThreads(Integer.parseInt(tmpStr));
    } else {
      setNumThreads(0);
    }

    tmpStr = Utils.getOption("numFeatTree", options);
    if ( tmpStr.length() != 0 ){
      m_numFeatTree = Integer.parseInt(tmpStr);
    } else {
      m_numFeatTree = 0;
    }

    setComputeImportances(Utils.getFlag("import", options));
    setComputeDropoutImportance(Utils.getFlag("importNew", options));
    setComputeInteractions(Utils.getFlag("interactions", options));
    setComputeInteractionsNew(Utils.getFlag("interactionsNew", options));

    super.setOptions(options);

    Utils.checkForRemainingOptions(options);
  }


  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */
  public Capabilities getCapabilities(){
    return new FastRandomTree().getCapabilities();
  }


  /**
   * Builds a classifier for a set of instances.
   *
   * @param data the instances to train the classifier with
   *
   * @throws Exception if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();

    // only class? -> build ZeroR model
    if(data.numAttributes() == 1){
      System.err.println(
        "Cannot build model (only class attribute present in data!), "
          + "using ZeroR model instead!");
      m_ZeroR = new weka.classifiers.rules.ZeroR();
      m_ZeroR.buildClassifier(data);
      return;
    }
    else{
      m_ZeroR = null;
    }

    /* Save header with attribute info. Can be accessed later by FastRfTrees
     * through their m_MotherForest field. */
    m_Info = new Instances(data, 0);

    m_bagger = new FastRfBagging();

    m_numAttributes = data.numAttributes();

    // Set up the tree options which are held in the motherForest.
    if(m_KValue > data.numAttributes() - 1) m_KValue = data.numAttributes() - 1;
    if(m_KValue < 1) m_KValue = (int)Utils.log2(data.numAttributes()) + 5;

    if(m_numFeatTree < 1) m_numFeatTree = (int) Math.pow(data.numAttributes(), 0.6) + 60; //(int) Math.sqrt(data.numAttributes()*2) + 60;
    if(m_numFeatTree >= data.numAttributes()) {
      m_numFeatTree = data.numAttributes() - 1;
      m_KValue = (int)Utils.log2(data.numAttributes()) + 1;
    }
    // Modify m_numFeatTree if we compute feature importance new
    if (this.getComputeDropoutImportance()) {
      // a minimum of 40 trees
      m_numTrees = Math.max(minTrees*2, m_numTrees);
      // a minimum of 20 trees with a specific attribute
      m_numFeatTree = Math.max(minTrees*data.numAttributes()/m_numTrees + 1, m_numFeatTree);
      // a minimum of 20 trees without a specific attribute
      m_numFeatTree = Math.min((m_numTrees - minTrees)*data.numAttributes()/m_numTrees, m_numFeatTree);
    }
    // Modify m_numFeatTree if we compute interactions new
    if (this.getComputeInteractionsNew()) {
      // a minimum of 40 trees
      m_numTrees = Math.max(40, m_numTrees);
      // half of the trees with a specific attribute
      m_numFeatTree = (m_numTrees/2)*data.numAttributes()/m_numTrees + 1;
    }

    FastRandomTree rTree = new FastRandomTree();
    rTree.m_MotherForest = this; // allows to retrieve KValue and MaxDepth
    // some temporary arrays which need to be separate for every tree, so
    // that the trees can be trained in parallel in different threads
    
    // set up the bagger and build the forest
    m_bagger.setClassifier(rTree);
    m_bagger.setSeed(m_randomSeed);
    m_bagger.setNumIterations(m_numTrees);
    m_bagger.setCalcOutOfBag(true);
    m_bagger.setComputeImportances( this.getComputeImportances() );
    m_bagger.setComputeDropoutImportance(this.getComputeDropoutImportance());
    m_bagger.setComputeInteractions(this.getComputeInteractions());
    m_bagger.setComputeInteractionsNew(this.getComputeInteractionsNew());

    m_bagger.buildClassifier(data, m_NumThreads, this);
    
  }


  /**
   * Returns the class probability distribution for an instance.
   *
   * @param instance the instance to be classified
   *
   * @return the distribution the forest generates for the instance
   *
   * @throws Exception if computation fails
   */
  public double[] distributionForInstance(Instance instance) throws Exception{

    if(m_ZeroR != null){  // default model?
      return m_ZeroR.distributionForInstance(instance);
    }

    return m_bagger.distributionForInstance(instance);

  }

  /**
   * Outputs a description of this classifier.
   *
   * @return a string containing a description of the classifier
   */
  public String toString(){

    StringBuilder sb = new StringBuilder();
    
    if(m_bagger == null)
      sb.append("FastRandomForest not built yet");
    else {
      sb.append("FastRandomForest of " + m_numTrees
        + " trees, each constructed while considering "
        + m_KValue + " random feature" + (m_KValue == 1 ? "" : "s") + ".\n"
        + "Out of bag error: " + Utils.doubleToString(m_bagger.measureOutOfBagError()*100.0, 3) + "%\n"
        + (getMaxDepth() > 0 ? ("Max. depth of trees: " + getMaxDepth() + "\n") : (""))
        + "\n");
      if ( getComputeImportances() ) {
        sb.append("Feature importances - increase in out-of-bag error (as % misclassified instances) after feature permuted:\n");
        double[] importances = new double[0];
        try {
          importances = m_bagger.getFeatureImportances();
        } catch (ExecutionException e) {
          e.printStackTrace();
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
        for ( int i = 0; i < importances.length; i++ ) {
          sb.append( String.format( "%d\t%s\t%6.4f%%\n", i+1, this.m_Info.attribute(i).name(),
                  i==m_Info.classIndex() ? Double.NaN : importances[i]*100.0 ) ); //bagger.getFeatureNames()[i] );
        }
      }
    }
    
    return sb.toString();
  }

  ////////////////////////////
  // Feature importances stuff
  ////////////////////////////

  // TODO Show warning or error while calling these methods without a suitable number of feature per tree

  /** @return the feature importances or <code>null</code> if the importances haven't been computed */
  public double[] getFeatureImportances() throws ExecutionException, InterruptedException {
    return m_bagger.getFeatureImportances();
  }

  public double[] getFeatureDropoutImportance() throws Exception {
    if (m_numFeatTree < minTrees*m_numAttributes/m_numTrees + 1) {
      throw new Exception("A given attribute appers in less than " + minTrees + " trees");
    }
    if (m_numFeatTree > (m_numTrees - minTrees)*m_numAttributes/m_numTrees) {
      throw new Exception("A given attribute appers in more than " + (m_numTrees - minTrees) + " trees");
    }
    return m_bagger.getFeatureDropoutImportance();
  }

  public double[][] getInteractions() throws ExecutionException, InterruptedException {
    return m_bagger.getInteractions();
  }

  public double[][] getInteractionsNew() throws ExecutionException, InterruptedException {
    return m_bagger.getInteractionsNew();
  }

  /**
   * @return compute feature importances?
   */
  public boolean getComputeImportances() {
    return m_computeImportances;
  }

  /**
   * @param computeImportances compute feature importances?
   */
  public void setComputeImportances(boolean computeImportances) {
    m_computeImportances = computeImportances;
  }

  /**
   * @return compute feature importances new?
   */
  public boolean getComputeDropoutImportance() {
    return m_computeDropoutImportance;
  }

  /**
   * @param computeImportances compute feature importances?
   */
  public void setComputeDropoutImportance(boolean computeImportances) {
    m_computeDropoutImportance = computeImportances;
  }

  /**
   * @return compute interactions?
   */
  public boolean getComputeInteractions() {
    return m_computeInteractions;
  }

  /**
   * @param computeInteractions compute interactions?
   */
  public void setComputeInteractions(boolean computeInteractions) {
    m_computeInteractions = computeInteractions;
  }

  /**
   * @return compute interactions new?
   */
  public boolean getComputeInteractionsNew() {
    return m_computeInteractionsNew;
  }

  /**
   * @param computeInteractionsNew compute interactions?
   */
  public void setComputeInteractionsNew(boolean computeInteractionsNew) {
    m_computeInteractionsNew = computeInteractionsNew;
  }

  ////////////////////////////
  // /Feature importances stuff
  ////////////////////////////

  /**
   * Main method for this class.
   *
   * @param argv the options
   */
  public static void main(String[] argv){
    runClassifier(new FastRandomForest(), argv);
  }

  public String getRevision(){
    return RevisionUtils.extract("$Revision: 2.0$");
  }

}


