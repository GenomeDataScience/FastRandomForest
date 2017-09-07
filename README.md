# FastRandomForest 2.0 beta

FastRandomForest is a re-implementation of the [Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm) classifier (RF) for the Weka machine learning environment. FastRF brings speed and memory use improvements over the original Weka RF, particulary for datasets with large numbers number of features or instances.

The current version, FastRF 2.0 beta, employs a particular algorithmic trick to improve efficiency over the standard Random Forest algorithm (as implemented in the previous FastRF 0.99 or as in Weka RF), while retaining the accuracy of predictions.
FastRF 2.0b was developed by [Jordi Piqué Sellés](https://www.linkedin.com/in/jordi-piqu%C3%A9-sell%C3%A9s-8b84baa5/) at the [Genome Data Science lab](https://www.irbbarcelona.org/en/research/genome-data-science) of the IRB Barcelona. The code is a much-improved version of [FastRF 0.99](https://code.google.com/archive/p/fast-random-forest/) (by Fran Supek), which is itself loosely based on the RF implementation in Weka 3.6.


## How does it work?
In the FastRF 2 algorithm each tree is built from a subset of attributes from the entire dataset.  In comparison, in the standard RF, individual nodes are constructed using subsets of attributes, but there are no tree-wise constraints.  An improvement in execution time using this trick can be substantial: **average 2.41-fold speed improvement over Weka RF** across 33 tested real-world datasets of intermediate to large size; 2.76-fold and 6.20-fold for synthetic datasets based on the RDG1 and BayesNet generators, respectively. More details on [speed benchmarks wiki page](https://github.com/jordipiqueselles/FastRandomForest/wiki/Results). 

Overall, we find that use of the FastRF 2.0 algorithmic trick retains the classification accuracy of the original Weka RF and FastRF 0.99 implementation.  One possible explanation for this is that sub-sampling attributes per tree helps decorrelate the predictions of the individual trees, which is a desirable property in an ensemble classifier.  Of note, for individual datasets the accuracy may vary in either direction - please see the [accuracy benchmarks wiki page](https://github.com/jordipiqueselles/FastRandomForest/wiki/Results).

## When is FastRF helpful?

FastRF 2 brings large benefits in speed (and to some extent memory use) over Weka RF with datasets that have:
*	A large number of instances
*	A lot of numeric attributes or a lot of binary categorical attributes
*	Attributes with missing values

FastRF 2 is less beneficial, or in extreme cases detrimental compared to Weka RF when datasets have:
*	A lot of multi-categorical attributes with ≥5 categories
*	Datasets stored in sparse format are not handled at all by FastRF
*	Regression is not yet implemented in FastRF 2.0 beta

Generally, the larger the dataset, the larger the gain in speed of FastRF 2.0 over the standard Weka RF.

## Miscellaneous 

### Parameter choice.
As with all RF implementations, FastRF 2.0beta is reasonably robust to choice of parameters.  More trees are generally desirable.  A user should not need to change the values of default values of `m_Kvalue` and `m_numFeatTree` parameters that control the number of features considered per node and per tree; these defaults may change in future FastRF versions. Details on the [Parameters FastRF wiki page](https://github.com/jordipiqueselles/FastRandomForest/wiki/Parameters).

### A caveat. 
Rarely, on some datasets, the class probabilities (predictions for each instance) in FastRF 2.0 beta might have a differently shaped distribution compared to Weka RF. That means that the use of the default 50% probability cutoff for calling the positive or negative class may result in different predictions across many instances (possibly increasing or decreasing the % correctly classified instances; see benchmarks page). Importantly, the AUC score remains similar, meaning that the classifiers overall have similar discrimination power. Bear in mind that the 50% cutoff might in some datasets not mean the same thing in FastRF 2.0beta as it does in Weka RF. We are working to better understand this; see [Future work Wiki page](https://github.com/jordipiqueselles/FastRandomForest/wiki/Future-work) for other issues of interest.

## Future perspectives.
The algorithmic trick that FastRF 2.0 employs enables a novel type of attribute importance measure to be computed, called it the _dropout importance_ (named by a distant analogy to the dropout trick used in deep learning). In such a forest where some trees are guaranteed not to have a particular attribute, we can compute the attribute importance analysing the out-of-bag (OOB) error of the trees that had access to this particular attribute versus the OOB error of the trees that did not have access this attribute.  

Warning -- this _dropout importance_ algorithm is highly experimental and untested.  It is likely to give very different results compared to standard attribute importance measures.  The dropout importance is meant to capture the unique contribution of each feature which cannot be provided by other features, and might therefore be useful in prioritizing causal relationships.


# How to use FastRF 2.0 beta

You need to add the Weka library and the FastRandomForest_1.0.jar file to your Java classpath in order to use FastRF 2.0 beta.

The following code shows how to create a Random Forest from a dataset and how to make predictions based on that forest.  This is rather similar to how other Weka classifiers are invoked from Java code; see the [Weka instructions page](http://weka.wikispaces.com/Use+WEKA+in+your+Java+code).


```java
// Remember that this code can throw exceptions

// Loading the instances
ConverterUtils.DataSource source = new ConverterUtils.DataSource("/some/where/data.arff");
Instances data = source.getDataSet();
if (data.classIndex() == -1) {
    data.setClassIndex(data.numAttributes() - 1);
}

// Creating the forest
FastRandomForest fastRandomForest = new FastRandomForest();
fastRandomForest.setOptions(new String[]{"-I", "500"}); // 500 trees
fastRandomForest.buildClassifier(data);

// Getting the OOB error
double error = fastRandomForest.measureOutOfBagError();

// Getting the prediction of the first instance (the probability that belongs to a certain class)
Instance firstInstance = data.get(0);
double[] prediction = fastRandomForest.distributionForInstance(firstInstance);
```

In case you have a dataset in CSV format, you can use the following code.

```java
CSVLoader loader = new CSVLoader();
loader.setSource(new File("/some/where/data.csv"));
Instances data = loader.getDataSet();
```

For more info and examples about how to use any classifier that extends the Weka's AbstractClassifier class, see Weka wiki http://weka.wikispaces.com/Use+WEKA+in+your+Java+code 


# Description of the folders

The folder **datasets** contains the datasets used for analysing the algorithm.

The folder **results** contains and Excel file with the execution time and accuracy of the Weka version, 
the FastRF 0.99 version and this new last version of FastRF (2.0 beta).

The folder **src/hr/irb/src** contains the source code of the project.

