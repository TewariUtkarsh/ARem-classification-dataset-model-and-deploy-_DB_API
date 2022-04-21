# Importing Libraries
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS, cross_origin
import operations
import db_oper
import mongo_oper
import json
import logging

# Creating Flask app instance
app = Flask(__name__)

logging.basicConfig(filename='test.log', level=logging.DEBUG, format="%(asctime)s  %(levelname)s: %(message)s")


@app.route('/', methods=['POST', 'GET'])
@cross_origin()
def index():
    """
    This function renders the index.html which extends base.html, when API is routed to the base URL.
    :return:
    Renders index.html page
    """

    try:
        logging.info("Rendering index.html")
        return render_template('index.html')
    except Exception as e:
        logging.error("Error while rendering index.html")
        logging.exception(f"Error while rendering index.html: {e}")
        print(f"Error while rendering index.html: {e}")


@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def predict():
    """
    This functions performs the following tasks:
    1. Gather input from user
    2. Create database and table for storing predicted vaulues
    3. Gather y_pred, confusion matrix, accuracy score
    4. Render result.html to display the final output
    """


    try:
        # Gather input from user
        x_list_input = []

        data = operations.data

        logging.info("Getting input from user")

        for i in data:
            x_list_input.append(float(request.form[i]))
        x = x_list_input
    except Exception as e:
        logging.error(f"Error while gathering User Input")
        logging.exception(f"Error while gathering User Input: {e}")
        print(f"Error while gathering User Input: {e}")


    try:
        # Gathering User Info for DataBase Process
        logging.info(f"Loading User info from user.json")
        user_info = json.load(open('user.json', 'r'))
    except Exception as e:
        logging.error("Error while loading user info from user.json")
        logging.exception(f"Error while gathering user info: {e}")
        print(f"Error while gathering user info: {e}")

    try:
        # Creating DB thus gathering conn and curr instances
        logging.info("Creating connection with database params for both MySQL and MongoDB")
        conn, curr = db_oper.create_db(host=user_info['host'],  database=user_info['database'], username=user_info['username'], passwd=user_info['passwd'])

        # Initializing connection for MongoDB client
        username = user_info['user']
        passwd = user_info['passwd']
        client = mongo_oper.init_conn(username=username, passwd=passwd)
        client.close()

    except Exception as e:
        logging.error("Error while Creating a connection")
        logging.exception(f"Error while Creating a connection: {e}")
        print(f"Error while Creating a connection: {e}")

    try:
        # Creating table to store predictions
        logging.info("Creating table for both MySQL and MongoDB")
        rows = operations.data
        conn = db_oper.create_table(conn=conn, curr=curr, tn=user_info['tn'], rows=rows, solvers=operations.solvers)


        # Creating a DB instance and creating a collection for the cluster of existing client
        db = mongo_oper.create_db(db_name=user_info['db_name'])
        coll = mongo_oper.create_coll(db=db, coll_name=user_info['coll_name'])

    except Exception as e:
        logging.error("Error while creating table")
        logging.exception(f"Error while Creating a Table: {e}")
        print(f"Error while Creating a Table: {e}")

    try:
        # Performing Standard Scaling on the User Input as our model is trained on Standard Scaled input
        logging.info("Performing Standard Scaling")
        x_list_input = operations.stnd_scale(x_list_input)
    except Exception as e:
        logging.error("Error while performing Standard Scaling")
        logging.exception(f"Error while Performing Standard Scaling to User Input: {e}")
        print(f"Error while Performing Standard Scaling to User Input: {e}")


    solvers = operations.solvers

    # Gathering Predictions from model
    y_list_pred = operations.predict(x_list_input)


    try:
        logging.info("Inserting prediction into table")
        # Inserting user input and predicted values in the table
        conn = db_oper.insert_table(conn=conn, curr=curr, tn=user_info['tn'], x=x, y=y_list_pred)


        # Inserting rows/document into existing cluster for the client
        doc = {}
        for i, j in zip(rows, range(len(rows))):
            doc[i] = x[j]

        for i, j in zip(solvers, range(len(solvers))):
            doc[i] = y_list_pred[j][0]

        mongo_oper.insert_doc(db_name=user_info['db_name'], coll_name=user_info['coll_name'], doc=doc)


    except Exception as e:
        logging.error("Error while inserting predicted values")
        logging.exception(f"Error while inserting predicted values to the table: {e}")
        print(f"Error while inserting predicted values to the table: {e}")

    try:
        logging.info("Gathering score and confusion matrix")
        scores = operations.score()
        confusion_matrices = operations.confusion_matrices()
    except Exception as e:
        logging.error("Error while gathering confusion matrix and accuracy score")
        logging.exception(f"Error while gathering confusion matrix and accuracy score: {e}")
        print(f"Error while gathering confusion matrix and accuracy score: {e}")

    try:
        logging.info("Rendering result.html")
        return render_template('result.html', solvers=solvers, y_list_pred=y_list_pred, scores=scores, confusion_matrices=confusion_matrices)
    except Exception as e:
        logging.error("Error while rendering result.html")
        logging.exception(f"Error while rendering result.html: {e}")
        print(f"Error while rendering result.html: {e}")


@app.route('/report', methods=['POST', 'GET'])
@cross_origin()
def report():
    """
    This function is responsible for Rendering/Generating the Pandas Profile report for our Dataset by rendering report.html
    """

    try:
        logging.info("Rendering report.html for Pandas Profile")
        return render_template("report.html")
    except Exception as e:
        logging.error("Error occurred while rendering Profile Report")
        logging.exception(f"Error occurred while rendering Profile Report: {e}")
        print(f"Error occurred while rendering Profile Report: {e}")


if __name__ == '__main__':
    try:
        app.run(debug=True)
        logging.info("Running app...")
    except Exception as e:
        logging.error("Error while running app")
        logging.exception(f"Error while running the app: {e}")
        print(f"Error while running the app: {e}")
