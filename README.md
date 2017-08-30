# FastRandomForest

FastRandomForest is a re-implementation of the [Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm) 
classifier (RF) for the Weka environment that brings speed and memory use improvements over the original Weka RF.



## Optimizations

### Preselection of features for each tree



### Change from entropy to gini

In this new version, the function used to evaluate a split is the gini impurity. In older versions, as in the
Weka implementation, the entropy it's used. This gives a speedup improvement, because the gini impurity doesn't need
to compute a logarithm, which is a slow function to calculate. 

However, the functions to calculate the entropy still exist in the SplitCriteria class file.

### Recalculating the gini score only between instances that belong to different classes

In a numerical attribute, you only need to recalculate the gini score when two consecutive instances belong to different
classes, because the maximum gain will never be between two instances of the same class.

## Future work

* Ability to support numerical class (regression tree)

* Change the one vs all split used for categorical attributes to a "nomal" split, the one that have many sons as 
different values has the categorical attribute. This should improve the execution time in datasets like 
**kdd_ipums_la_97-small.arff**, where there's only categorical attributes with lots of different values.

* Test accurately the functions used to calculate feature importance and feature interaction.

