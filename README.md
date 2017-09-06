# FastRandomForest

FastRandomForest is a re-implementation of the [Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm) 
classifier (RF) for the Weka environment that brings speed and memory use improvements over the original Weka RF.
It is specially efficient with datasets with a huge number of features.

This new version has a different approach. Each tree doesn't have all the features in the dataset, but it has
only a subset of this features. This trick reduces dramatically the execution time in datasets where there are
a lot of different attributes. However, someone could think that the accuracy is lower. The fact is the accuracy
is practically the same as in the last version of FRF (FRF 0.99) and the latest version of the Weka implementation
of RF. That's because we have also increased a little bit the number of attributes analyzed in each node to
make the split.

That trick also opens a new door for computing feature importance and feature interaction. In a forest where
each tree doesn't have all the attributes, we can compute the feature importance analysing the OOB error of 
the trees that have a specific feature vs the OOB error of the trees that doesn't have this feature.

One of the major challenges of that new implementation of Random Forest was choosing a suitable value for the
number of attributes analysed in each node (`m_KValue`) and the number of attributes for each tree 
(`m_numFeatTree`). Two generators from Weka, the RDG1 and the BayesNet, were used to generate datasets with
different number of instances (from 100 to 25600) and different number of attributes (from 100 to 25600).
The values that were found are `m_KValue = log2(numAttributes) + 5` and 
`m_numFeatTree = pow(numAttributes, 0.6) + 60`. These values give us a better execution time without compromising
the accuracy. However, a deep analysis of the behaviour of the forest when varying these two parameters is
needed. We believe that these parameters can be modified in order to improve more the execution time and the
accuracy.

If you want to see the comparison of the accuracy and execution time between FRF_1.0, FRF_0.99 and the RF Weka you can
go to the [results](https://github.com/jordipiqueselles/FastRandomForest/wiki/Results) page.

For farther explanations and details of the project you can visit the 
[project wiki](https://github.com/jordipiqueselles/FastRandomForest/wiki).

If you wont to visit the old repository of FastRandomForest you can follow this link:

[https://code.google.com/archive/p/fast-random-forest/](https://code.google.com/archive/p/fast-random-forest/)

# How to use it

You need to add the Weka library and the FastRandomForest_1.0.jar file in your project in order to use FastRandomForest.

The following code shows how to create a Random Forest from a dataset and how to make predictions based on that 
forest.

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

For more info and examples about how to use any classifier that extends the Weka's AbstractClassifier class,
you can follow this link: 
[http://weka.wikispaces.com/Use+WEKA+in+your+Java+code](http://weka.wikispaces.com/Use+WEKA+in+your+Java+code)

# Description of the folders

The folder **datasets** contains the datasets used for analysing the algorithm.

The folder **results** contains CSV files with the execution time and accuracy of the Weka version, 
the FRF 0.99 version and this new last version of FRF (1.0).

The folder **src/hr/irb/src** contains the source code of the project.

