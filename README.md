# FastRandomForest

FastRandomForest is a re-implementation of the Random Forest classifier (RF) for the Weka environment that brings speed 
and memory use improvements over the original Weka RF.



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

