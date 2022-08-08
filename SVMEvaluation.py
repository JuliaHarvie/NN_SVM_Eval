import pandas as pd
import numpy as np
import argparse
import sys
import glob
import os
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics

from datetime import datetime

now = datetime.now()
start_time = now.strftime("%H:%M:%S")

### Setting up arguments, specify k for k-fold cross validation
argparser = argparse.ArgumentParser() 
argparser.add_argument("k",
		help="Number of folds to be used for cross validation", type=int)

args = argparser.parse_args(sys.argv[1:])
k = args.k

if os.path.exists("SVM_confusion.txt"):
	os.remove("SVM_confusion.txt")
if os.path.exists("SVM_accuracy.txt"):
	os.remove("SVM_accuracy.txt")    

### Model is validated, trained and test for every fold. The results used to build a 
#   confusion matrix and calculate an accuracy score for each fold. This is repeated 
#   for every fold and the resulting matrices and accuracy scores are exported as text 
#   files to be used for further anlysis.

for f in range(0,k):
# Load in all data and seperate features from reponse values (x and y respectively)
    print(f"Reading in data for fold {f+1}")
    for x in glob.glob("Datasets/*.csv"):
        if f'{f}' in x and "test" in x:
            testing_data = pd.read_csv(x)
    testing_data = testing_data.to_numpy()
    y_testing_data = np.ravel(testing_data[:,0])
    x_testing_data = testing_data[:,1:]

    for x in glob.glob("Datasets/*.csv"):
        if f'{f}' in x and "train" in x:
            training_data = pd.read_csv(x)
    training_data = training_data.to_numpy()
    y_training_data = np.ravel(training_data[:,0])
    x_training_data = training_data[:,1:]

    for x in glob.glob("Datasets/*.csv"):
        if f'{f}' in x and "validation" in x:
            validation_data = pd.read_csv(x)
    validation_data = validation_data.to_numpy()
    y_validation_data = np.ravel(validation_data[:,0])
    x_validation_data = validation_data[:,1:] 

# Use the validation data and the sk learn function HalvingGridSearch
# to determine best parameters to use for model fitting
    parameters = [
# Regulization parameter        
        {'C': [1, 10, 100, 1000] }]     
    print("Performing grid search")
    svc_search = HalvingGridSearchCV(LinearSVC(), parameters, verbose = 3).fit(x_validation_data, y_validation_data)
    best_parameters = svc_search.best_params_
    print("Best parameters found for SVN", best_parameters)

    print("Creating SVC model to be trained")
    model = LinearSVC(**best_parameters)
    model.fit(x_training_data, y_training_data)

    print(" . Performing evaluation:")
    y_predict = model.predict(x_testing_data)
    prediction_accuracy = metrics.accuracy_score(y_testing_data, y_predict)

    print(f'Performance accuracy: {prediction_accuracy}')
# Write the accuracy to a text file
    f = open('SVM_accuracy.txt', 'a')
    f.write("%s\n" % prediction_accuracy)
    f.close()
# Write the confusion matrix to a text file
    cm = metrics.confusion_matrix(y_testing_data, y_predict)
    with open('SVM_confusion.txt', 'a') as f:
        f = open('SVM_confusion.txt', 'a')
        f.write("%s \n" % cm) 
        f.close()
 

now = datetime.now()
end_time = now.strftime("%H:%M:%S")

print("Start time: ", start_time)
print("End time: ", end_time)