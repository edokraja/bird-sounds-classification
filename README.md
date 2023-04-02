# bird-sounds-classification
Python project classifying different bird sounds

# Birds
Under the file ./description/class_names.txt
0: other
1: comcuc
2: cowpig1
3: eucdov
4: eueowl1
5: grswoo
6: tawowl1

# Feature Names
584 pre computed features under ./description/feature_names.txt


# Task 3
TODO: 
* Concatenate all matrices of all birds row-wise
* Re-write task 3 to fit the new numpy array of 120k*548

# Task 4
Todo:
* Add the label column to the original (concatenated) dataframe respectively
* Use corr-based feature selection model (for now) to answer both
  * Which features are good for classification? use variance for each feature
  * Which features are highly correlated with the label column