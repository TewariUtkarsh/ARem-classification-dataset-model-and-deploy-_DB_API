# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import pickle
import json
import logging

'''
# accuracy_score = model.score
# roc_auc_score(y_true, y_pred) = auc(fpr, tpr)
'''

try:
    # Loading Dataset
    df = pd.read_csv('final_dataset.csv')
except Exception as e:
    print(f"Error occurred while loading dataset: {e}")

try:
    # Feature and Label Selection
    classes = ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']
    y = df['label']
    x = df.drop('label', axis=1)

except Exception as e:
    print(f"Error while selecting Features and Label: {e}")

try:
    # Standardize x (Features) using StandScaler
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    std_x = scaler.transform(x)
except Exception as e:
    print(f"Error occurred while Standardizing: {e}")

try:
    # Train and Test Split
    x_train, x_test, y_train, y_test = train_test_split(std_x, y, test_size=.25, random_state=200)
except Exception as e:
    print(f"Error while train_test_split: {e}")

try:
    # Creating and Training Logistic Regression Model:
    solvers = ['lbfgs', 'liblinear', 'newton_cg', 'sag', 'saga']

    # 1. Using lbfgs solver and L2 penalty:
    lr_lbfgs = LogisticRegression(solver='lbfgs', multi_class='ovr', penalty='l2')
    lr_lbfgs.fit(x_train, y_train)

    # 2. Using liblinear solver and L2 penalty:
    lr_liblinear = LogisticRegression(solver='liblinear', multi_class='ovr', penalty='l2')
    lr_liblinear.fit(x_train, y_train)

    # 3. Using newton-cg solver and L2 penalty:
    lr_newton_cg = LogisticRegression(solver='newton-cg', multi_class='ovr', penalty='l2')
    lr_newton_cg.fit(x_train, y_train)

    # 4. Using sag solver and L2 penalty:
    lr_sag = LogisticRegression(solver='sag', multi_class='ovr', penalty='l2')
    lr_sag.fit(x_train, y_train)

    # 5. Using saga solver and L2 penalty:
    lr_saga = LogisticRegression(solver='saga', multi_class='ovr', penalty='l2')
    lr_saga.fit(x_train, y_train)

    # Storing all the models in a list
    models = [lr_lbfgs, lr_liblinear, lr_newton_cg, lr_sag, lr_saga]
except Exception as e:
    print(f"Error occurred while creating models: {e}")

try:
    # Saving/Dumping every model with different params
    pickle.dump(lr_lbfgs, open('lr_lbfgs.pkl', 'wb'))
    pickle.dump(lr_liblinear, open('lr_liblinear.pkl', 'wb'))
    pickle.dump(lr_newton_cg, open('lr_newton_cg.pkl', 'wb'))
    pickle.dump(lr_sag, open('lr_sag.pkl', 'wb'))
    pickle.dump(lr_saga, open('lr_saga.pkl', 'wb'))
except Exception as e:
    print(f"Error occurred while dumping models: {e}")


try:
    # Gathering and Storing the Accuracy Scores of each solver in dict
    accuracy = {}
    for s, m in zip(solvers, models):
        accuracy[s] = m.score(x_test, y_test)
except Exception as e:
    print(f"Error occurred while gathering the accuracy score for model(solver): {e}")


try:
    # Generating the Confusion Matrix of each solver and storing it in a list
    confusion_matrices = {}
    for s, m in zip(solvers, models):
        confusion_matrices[s] = confusion_matrix(y_test, m.predict(x_test))
except Exception as e:
    print(f"Error occurred while generating Confusion Matrix: {e}")


# Getting the ROC curve(fpr, tpr, threshold) and ROC AUC score for all the solvers and plotting and storing their graphs
try:
    for s, m in zip(solvers, models):

        # Binarize the labels for ovr(1,0,0,0,0,0,0 format)
        y_bin_train = label_binarize(y_train, classes=classes)
        y_bin_test = label_binarize(y_test, classes=classes)

        # Creating a OneVsResClassifier with LogisticRegression classifier
        lr_ovr = OneVsRestClassifier(estimator=m)

        # Fitting our OVR model
        lr_ovr.fit(x_train, y_bin_train)

        # Storing the Predicted class(here predicted values are binarized)
        y_bin_pred = lr_ovr.predict(x_test)

        fpr = {}
        tpr = {}
        thresh = {}
        roc_auc = {}
        n = len(set(y))

        # Getting the fpr, tpr and threshold values for individual classes for a particular solver
        for i in range(n):
            fpr[i], tpr[i], thresh[i] = roc_curve(y_bin_test[:, i], y_bin_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plotting the graph with respective fpr, tpr and threshold for every classes individually
        colours = ['red', 'yellow', 'blue', 'black', 'purple', 'green', 'orange']
        plt.subplots(figsize=(20, 20))

        for i in range(n):
            plt.xlabel = 'FPR'
            plt.ylabel = 'TPR'
            plt.title = f'ROC Curve({s})'
            plt.plot(fpr[i], tpr[i], color=colours[i], label=f'ROC(Class:{i})', marker='o')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label=f'OVR(Class:{i})(area = %0.2f)' % roc_auc[i])
            plt.legend()

        # Saving the Graph for every solver.
        plt.savefig(f'{s}_plot')
except Exception as e:
    print(f"Error occurred while generating ROC curve for models(solver): {e}")


try:
    # Storing Confusion matrix and Accuracy scores in .npy file for future reference(not using json files as ndarray is not json serialized)
    # json.dump(obj=[confusion_matrices, accuracy], fp=open('info.json', 'w'))
    final = np.array((confusion_matrices, accuracy))
    np.save('info.npy', final, allow_pickle=True)
except Exception as e:
    print(f"Error occurred while saving confusion matrix and accuracy score in .npy file: {e}")




'''
# Performed - Done
# try:
#     # Performing Cross Validation for Hyperparameter tuning
#     """
#     Output:
#
#     """
#     grid_params = {
#         "penalty": ['l1', 'l2', 'elasticnet', None],
#         "C": [0, .5, 1, 1.5, 2],
#         "solver": ['lbfgs', 'liblinear', 'newton_cg', 'sag', 'saga'],
#         "max_iter": [100, 1000, 10000, 100000, 1000000],
#         "multi_class": ['auto', 'ovr', 'multinomial']
#     }
#     lr_gridCV = GridSearchCV(estimator=LogisticRegression(), param_grid=grid_params, cv=10, verbose=1)
#     lr_gridCV.fit(x_train, y_train)
#
#     # Saving the GridSearchCV object
#     pickle.dump(lr_gridCV, open('lr_gridCV.sav', 'wb'))
# except Exception as e:
#     print(f"Error while Performing Cross Validation: {e}")
'''

'''
# 
# try:
#     # Creating a model with GridSearchCV params
#     lr_best = LogisticRegression(penalty=, C=, solver=, max_iter=, multi_class=)
#     lr_best.fit(x_train, y_train)
#     accuracy['lr_cv'] = lr_best.score(x_test, y_test)
#     confusion_matrices.append(confusion_matrix(y_test, lr_best.predict(y_test)))
# except Exception as e:
#     print(f"Error while Generating Model: {e}")
# 
# # Using in built LogisticRegressionCV
'''

"""
Conclusion:
LogisticRegressionCV is similar to GridSearchCV just that the default solver in LogisticRegressionCV is different
"""
