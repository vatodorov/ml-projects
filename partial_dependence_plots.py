#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created: September 21, 2018

@author: valentint
"""

data_path = "/Users/valentint/Documents/Data/Carvana/training.csv"
sample_size = 0.1
test_size = 0.33
target = "isbadbuy"


# Import the modeling utilities
import sys
sys.path.insert(0, "/Users/valentint/Documents/Projects/ModelTools/UtilityFunctions")
import modeling_utilities as utils

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Read the data and change the columns to lowercase
df = pd.read_csv(data_path)
df.columns = [x.lower() for x in list(df.columns)]
df.head(5)


# Frequencies - decide if I need to create groups
# Just drop these fields for now
pd.crosstab(df["auction"], columns = "count")
pd.crosstab(df["make"], columns = "count")
pd.crosstab(df["wheeltype"], columns = "count")
pd.crosstab(df["color"], columns = "count")
pd.crosstab(df["vnst"], columns = "count")
pd.crosstab(df["size"], columns = "count")

# Fix lowercase for "Manual"
pd.crosstab(df["transmission"], columns = "count")
df["transmission"] = np.where(df["transmission"] == "Manual", "MANUAL", df["transmission"])


# One-hot encode variables
df = pd.concat([df, pd.get_dummies(df["auction"])], axis = 1)
df = pd.concat([df, pd.get_dummies(df["transmission"])], axis = 1)
df = pd.concat([df, pd.get_dummies(df["wheeltype"])], axis = 1)

df.columns = [x.lower() for x in list(df.columns)]


# Drop variables
drop_vars = ["refid", "purchdate", "model", "make", "trim", "submodel", "color",
            "transmission", "wheeltype", "nationality", "primeunit", "aucguart",
            "vnzip1", "vnst", "size", "topthreeamericanname", "auction"]
df.drop(drop_vars, axis = 1, inplace = True)

# Remove all rows with NAs
df.dropna(inplace = True)

# Downsample the data
analysis_set = df.sample(frac = sample_size,
                         replace = False,
                         weights = None,
                         random_state = 7894)

# Create samples for training and testing
x_train, x_test, y_train, y_test = train_test_split(analysis_set.drop(target, axis = 1),
                                                    analysis_set[target],
                                                    test_size = test_size,
                                                    random_state = 4567)

# Estimate a random forest model
# Use optimal parameters to estimate a Random Forest model
rf_params = {"n_estimators": 300,
             "max_depth": 10,
             "n_jobs": -1,
             "random_state": 4561,
             "verbose": 0,
             "criterion": "gini",
             "class_weight": "balanced"}

model = RandomForestClassifier(**rf_params)
model.fit(x_train, y_train)

# Predict on training set and check the accuracy
utils.modelAccuracyStats(model, "Random Forest", x_train, y_train, x_test, y_test)

# Modify the data frame with the features ranking
rf_features = pd.DataFrame(data = model.feature_importances_.reshape(-1, len(model.feature_importances_)),
                           columns = list(x_train))
rf_features = rf_features.stack().to_frame()
rf_features.reset_index(inplace = True)
rf_features.rename(columns = {"level_0": "id", "level_1": "features", 0: "estimate"}, inplace = True)
rf_features.drop(["id"], axis = 1)
rf_features.sort_values("estimate", ascending = True, inplace = True)

# Plot the most important features
rf_var_imp = rf_features.tail(50)
utils.plot_variable_importance(rf_var_imp, "features", "estimate")


######################################

# Infer the type of variables or provide a list for categorical, continous, and binary
# Returns - categorical (1, 2, 3, 4, 10, 13), dummy (0 or 1), binary (4 or 5), continous
# The output is a dictionary - keys are the types, and the values are lists with the features
features_type = utils.check_feature_type(x_test,
                                         categorical_threshold = 10,
                                         categorical_vars = ["vehyear", "test_categorical"],
                                         continuous_vars = ["test_continuous"])

# Create a dataframe with the default values for each feature
# dummy - 1
# continuous - average
# categorical - median
# binary - the smaller value
# TO DO: Change the default for binary to the higher level - e.g. 5 in the case of [4, 5]

features_default = dict()
for k in features_type.keys():
    if k == "binary":
        features = features_type["binary"]
        for i in range(len(features)):
            features_default.update({features[i]: df[features[i]].mean()})
    elif k == "dummy":
        features = features_type["dummy"]
        for i in range(len(features)):
            features_default.update({features[i]: [1]})
    elif k == "continuous":
        features = features_type["continuous"]
        for i in range(len(features)):
            features_default.update({features[i]: df[features[i]].mean()})
    elif k == "categorical":
        features = features_type["categorical"]
        for i in range(len(features)):
            features_default.update({features[i]: df[features[i]].median()})

# Create dataframe for scoring
scoring_df = pd.DataFrame(features_default)

# # Iterate over each feature, change the feature values and score every time
# Keep the deault values for the other features unchanged
#   Binary: iterate for each of the two categories
#   Dummy: iterate for 0 & 1
#   Continuous: iterate in steps of 5 percentiles
#   Categorical: iterate by the categorical values

# Create a dataframe where the columns are each of the continuous features, and the values are their percentiles
# TO DO: For now we'll use the median for continuous, but may change it to mean
for k in features_type.keys():
    if k == "continuous":
        percentile_ranges = [round(x, 2) for x in np.linspace(0, 1, num = 25, endpoint = False).tolist()]
        features_percentiles = df[features_type["continuous"]].describe(percentiles = percentile_ranges)
        features_percentiles.drop(["count", "mean", "std", "min"], inplace = True)


# Loop through the feature types - dummy, continuous, categorical, binary
feature = []
feature_type = []
value = []
proba = []

for k in features_type.keys():
    # Loop through each dummy and calculate the predicted probability for 0 and 1
    if k == "dummy":
        print ("Calculating predictions for the dummy features ... \n")
        for v in features_type[k]:
            for i in range(2):
                scoring_dfc = scoring_df.copy()
                scoring_dfc[v] = i
                feature.append(v)
                feature_type.append(k)
                value.append(i)
                proba.append(model.predict_proba(scoring_dfc)[:, 1][0])

    # Loop through the values of each feature from the features_percentiles data frame
    # Calculate the predicted probability for each 10% increment in the value of a feature
    elif k == "continuous":
        print ("Calculating predictions for the continuous features ... \n")
        for v in features_type[k]:
            f = features_percentiles[v]
            for i in range(len(f)):
                scoring_dfc = scoring_df.copy()
                scoring_dfc[v] = f[i]
                feature.append(v)
                feature_type.append(k)
                value.append(f[i])
                proba.append(model.predict_proba(scoring_dfc)[:, 1][0])
                
    # Loop through the values of the categorical features
    elif k == "categorical":
        print ("Calculating predictions for the categorical features ... \n")
        for v in features_type[k]:                                          # ['vehyear', 'test_categorical', 'wheeltypeid', 'vehicleage']
            feature_values = list(df[v].value_counts().index)               # example: list(df['vehyear'].value_counts().index)
            for values in feature_values:                                   # values - 2006, 2005, 2007, 2008, 2009, etc.
                scoring_dfc = scoring_df.copy()
                scoring_dfc[v] = values
                feature.append(v)
                feature_type.append(k)
                value.append(values)
                proba.append(model.predict_proba(scoring_dfc)[:, 1][0])

    # Loop through the values of the binary features
    elif k == "binary":
        print ("Calculating predictions for the binary features ... \n")
        for v in features_type[k]:
            feature_values = list(df[v].value_counts().index)
            for values in feature_values:
                scoring_dfc = scoring_df.copy()
                scoring_dfc[v] = values
                feature.append(v)
                feature_type.append(k)
                value.append(values)
                proba.append(model.predict_proba(scoring_dfc)[:, 1][0])


# Create a dataframe from the dictionary
print ("Combine the predictions in a data frame and plot")
predictions = pd.DataFrame({"feature": feature,
                            "feature_type": feature_type,
                            "value": value,
                            "proba": proba})


# Plot features
keep_type = "continuous"
df_sub = predictions[predictions["feature_type"] == keep_type]
df_sub["feature"].value_counts()


keep_feature = "mmracquisitionretailaverageprice"
df_sub2 = df_sub[df_sub["feature"] == keep_feature]

x = df_sub2["value"]
series = df_sub2[["proba"]].to_dict("list")
utils.plot_multiple_graphs(graph_type = "line", x = x, y_dict = series, xlabel = "Values", ylabel = "Predicted Probability", graph_columns = 3)
