# Importing Libraries
import operations
import mysql.connector as connection
import logging

logging.basicConfig(filename='test.log', level=logging.DEBUG, format="%(asctime)s  %(levelname)s:  %(message)s")

solvers = operations.solvers


def create_conn(host, database, username, passwd):
    """
    This function is responsible for initiating a connection for MySQL.
    :param host: name of the host
    :param database: name of the existing database
    :param username: username for the connection
    :param passwd: password for the connection
    :return: Returns connection instance after initiating connection.
    """
    try:
        logging.info("Initiating Connection....")
        conn = connection.connect(host=host, database=database, user=username, passwd=passwd, use_pure=True)

    except Exception as e:
        logging.error(f"Error while initiating connection")
        logging.exception(f"Error while initiating connection: {e}")
        print(f"Error while initiating connection: {e}")
    else:
        return conn


def create_curr(conn):
    """
    This function is responsible for generating a cursor instance for an existing MySQL connection.
    :param conn: connection instance
    :return: cursor instance for existing connection
    """
    try:
        logging.info("Creating cursor instance for existing connection")
        curr = conn.cursor(buffered=True)
    except Exception as e:
        logging.error(f"Error while generating cursor instance for the MySQL connection")
        logging.exception(f"Error while generating cursor instance for the MySQL connection: {e}")
        print(f"Error while generating cursor instance for the MySQL connection: {e}")
    else:
        return curr


def create_db(host, database, username, passwd):
    """
    This function is responsible for creating DB
    :param host: name of the host
    :param database: name of an existing database
    :param username: username for the connection
    :param passwd: password for the connection
    :return: connection instance and cursor instance
    """
    try:
        logging.info("Creating Database")
        conn = create_conn(host, database, username, passwd)
        curr = create_curr(conn)
    except Exception as e:
        logging.error(f"Error while creating DB")
        logging.exception(f"Error while creating DB: {e}")
        print(f"Error while creating DB: {e}")
    else:
        return conn, curr


def create_table(conn, curr, tn, rows, solvers):
    """
    This function is responsible for creating a table
    :param conn: connection instance
    :param curr: cursor instance for existing connection
    :param tn: table name
    :param rows: rows
    :param solvers: solver names
    :return:
    """
    try:
        logging.info("Creating Table")
        curr.execute(f"create table {tn}( {rows[0]} float, {rows[1]} float,{rows[2]} float,{rows[3]} float,{rows[4]} float,{rows[5]} float, {solvers[0]} varchar(255), {solvers[1]} varchar(255), {solvers[2]} varchar(255), {solvers[3]} varchar(255), {solvers[4]} varchar(255));")
    except Exception as e:
        logging.error(f"Error while creating table")
        logging.exception(f"Error while creating table: {e}")
        print(f"Error while creating table: {e}")
    else:
        return conn


def insert_table(conn, curr, tn, x, y):
    """
    This function is responsible for inserting data to the table
    :param conn: connection instance
    :param curr: cursor instance for existing connection
    :param tn: table name
    :param x: x values
    :param y: y values
    :return: Returns connection instance after commit insertion
    """
    try:
        logging.info("Inserting predicted result in the table")
        curr.execute(f"insert into {tn} values( {x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}, {x[5]}, '{y[0][0]}', '{y[1][0]}', '{y[2][0]}', '{y[3][0]}', '{y[4][0]}' );")
        conn.commit()
    except Exception as e:
        logging.error(f"Error while inserting data to the table")
        logging.exception(f"Error while inserting data to the table: {e}")
        print(f"Error while inserting data to the table: {e}")
    else:
        return conn
