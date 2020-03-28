"""
download mongodb
create the following directories for your project
data
data/db
data/logpython

must use 127.0.0.1 on windows
pip install pymongo

"""
import os
import json
import pandas as pd
from pymongo import MongoClient



class MongoDBConnection(object): # pylint: disable=R0205
    """MongoDB Connection"""
    def __init__(self, host='127.0.0.1', port=27017):
        self.host = host
        self.port = port
        self.connection = None

    def __enter__(self):
        self.connection = MongoClient(self.host, self.port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

def import_data(directory_name, product_file, customer_file, rentals_file):
    """function to import data from csv files"""
    os.chdir(directory_name)
    products = pd.read_csv(product_file)
    customers = pd.read_csv(customer_file)
    rentals = pd.read_csv(rentals_file)
    rental_customer = customers.set_index('user_id').join(rentals.set_index('user_id'))
    return products, customers, rentals, rental_customer

def create_mongo():
    """function to create MongoDB"""
    mongo = MongoDBConnection()
    with mongo:
        database = mongo.connection.media
        product = database["product"]
        prod = PRODUCT.to_json(orient='records')
        json_prod = json.loads(prod)
        product.insert_many(json_prod)
        database = mongo.connection.media
        customer = database["customer"]
        cus = CUSTOMER.to_json(orient='records')
        json_cus = json.loads(cus)
        customer.insert_many(json_cus)
        database = mongo.connection.media
        rental = database["rentals"]
        ren = RENTAL.to_json(orient='records')
        json_ren = json.loads(ren)
        rental.insert_many(json_ren)
    return print("mongoDB created")

def show_available_products():
    """function to show all available products"""
    print("These are the available products \n", PRODUCT)

def show_rentals(product_id):
    """function to return all rentals of a specific product_id"""
    output = RENTAL_CUSTOMER.loc[RENTAL_CUSTOMER['product_id'] == product_id]
    return output

if __name__ == "__main__":
    DIRECTORY_IMPORT = os.getcwd()
    CUSTOMER_IMPORT = "customer_file.csv"
    PRODUCT_IMPORT = "product_file.csv"
    RENTALS_IMPORT = "rentals_file.csv"
    PRODUCT, CUSTOMER, RENTAL, RENTAL_CUSTOMER = import_data(DIRECTORY_IMPORT
                                                             , PRODUCT_IMPORT,
                                                             CUSTOMER_IMPORT,
                                                             RENTALS_IMPORT)
    create_mongo()
    show_available_products()
    OUTPUT = show_rentals(1)
    print(OUTPUT)
