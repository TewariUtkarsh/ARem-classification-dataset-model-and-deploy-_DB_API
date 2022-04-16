# Importing Libraries
import pickle
import numpy as np
import pandas as pd
import logging

logging.basicConfig(filename='test.log', level=logging.DEBUG, format="%(asctime)s  %(levelname)s:  %(message)s")

# Required Data
data = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
solvers = ['lbfgs', 'liblinear', 'newton_cg', 'sag', 'saga']
models = ['lr_lbfgs', 'lr_liblinear', 'lr_newton_cg', 'lr_sag', 'lr_saga']
classes=['bending1', 'bending2', 'cycling', 'lying', 'sitting','standing','walking']


def stnd_scale(x):
    """
    This function is responsible for performing Standard Scaling on data
    :param x: Data to be standard scaled
    :return: Standard Scaled data
    """
    try:
        logging.info("Performing Standard Scaling from operations.py")
        scaler = pickle.load(open('scaler.pickle', 'rb'))
        std_x = scaler.transform([x])
    except Exception as e:
        logging.error(f"Error while Standard Scaling Data")
        logging.exception(f"Error while Standard Scaling Data: {e}")
        print(f"Error while Standard Scaling Data: {e}")
    else:
        logging.info("Standard Scaling successful.")
        return std_x


def predict(x):
    """
    This function is responsible for performing prediction on the input data
    :param x: Input data for Prediction
    :return: Predicted values
    """
    try:
        logging.info("Initializing Prediction from operations.py")
        y_list = []
        for i in models:
            m = pickle.load(open(f'{i}.pkl', 'rb'))
            y_pred = m.predict(x)
            y_list.append(y_pred)
    except Exception as e:
        logging.error(f"Error while predicting")
        logging.exception(f"Error while predicting: {e}")
        print(f"Error while predicting: {e}")
    else:
        logging.info("Prediction action performed successfully")
        return y_list


def score():
    """
    This function is responsible for loading model's accuracy score from info.npy
    :return: List of Accuracy Scores for all the Solvers
    """
    try:
        logging.info("Gathering Accuracy score for all solvers")
        nf = np.load('info.npy', allow_pickle=True)
        s = list(nf[1].values())
    except Exception as e:
        logging.error(f"Error while gathering accuracy score")
        logging.exception(f"Error while gathering accuracy score: {e}")
        print(f"Error while gathering accuracy score: {e}")
    else:
        logging.info("Accuracy Scored gathered successfully")
        return s


# confusion matrix: npy
def confusion_matrices():
    """
    This function is responsible for loading Confusion Matrix for every Solver
    :return: List of Confusion Matrix(Here each confusion matrix is converted from np.array to df to html code for tabular representation)
    """
    try:
        logging.info("Gathering Confusion Matrix from info.npy")
        cmf = np.load('info.npy', allow_pickle=True)
        cm = list(cmf[0].values())
        final = []
        for i in cm:
            df = pd.DataFrame(i, columns=classes)
            df.index = classes
            s = df.to_html().replace('\n', '')
            final.append(s)
    except Exception as e:
        logging.error(f"Error while gathering confusion matrix")
        logging.exception(f"Error while gathering confusion matrix: {e}")
        print(f"Error while gathering confusion matrix: {e}")
    else:
        return final
