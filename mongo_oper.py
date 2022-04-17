import pymongo
import operations
import urllib.parse
import logging
import json

user_info = json.load(open('user.json', 'r'))


def init_conn(username, passwd):
    p = urllib.parse.quote(passwd)
    client = pymongo.MongoClient(f"mongodb+srv://{username}:" + p + "@cluster0.vp3lc.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    return client


def create_db(db_name):
    client = init_conn(username=user_info['user'], passwd=user_info['passwd'])
    db = client[db_name]
    client.close()
    return db


def create_coll(db, coll_name):
    coll = db['coll_name']
    return coll


def insert_doc(db_name, coll_name, doc):
    # record = doc
    # client = init_conn(username=user_info['user'], passwd=user_info['passwd'])
    db = create_db(db_name=db_name)
    coll = create_coll(db=db, coll_name=coll_name)
    coll.insert_one(doc)



