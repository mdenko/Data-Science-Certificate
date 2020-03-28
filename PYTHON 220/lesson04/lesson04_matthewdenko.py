#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 10:06:01 2020

@author: matthewdenko
"""
import random
from random import randint
import pandas as pd
import numpy as np



def create_data():
    """function to generate random data set"""
    data_set = pd.DataFrame()
    customer_id = list()
    for i in range(1, 10001):
        customer_id.append(i)
        data_set = pd.DataFrame()
        data_set.loc[:, 'customer_id'] = np.array(customer_id)
    product_name = ('dining chair', 'dining table', 'bed', 'dining set',
                    'stool', 'couch', 'occasional table',
                    'recliner')
    product_name_random = random.choices(product_name, k=10000)
    data_set.loc[:, 'product_name'] = np.array(product_name_random)
    quantity_rented = (1, 2, 3, 4)
    quantity_rented_random = random.choices(quantity_rented, k=10000)
    data_set.loc[:, 'quantity_rented'] = np.array(quantity_rented_random)
    unit_rental_price_monthly = list()
    for i in range(0, 10000):
        unit_rental_price_monthly.append(random.uniform(1.5, 25))
    data_set.loc[:, 'unit_rental_price'] = np.array(unit_rental_price_monthly)
    rental_period_months = list()
    for i in range(0, 10000):
        rental_period_months.append(randint(6, 60))
    data_set.loc[:, 'rental_period_months'] = np.array(rental_period_months)
    return data_set


def top_products(data_frame):
    """function to return top products"""
    data_frame.loc[:, 'total_payment'] = (data_frame['unit_rental_price']
                                          * data_frame['quantity_rented']
                                          * data_frame['rental_period_months'])
    data_set = data_frame.groupby(['product_name']).agg({'total_payment': 'sum'})
    data_set = data_set.nlargest(10, 'total_payment')
    return data_set


def top_five_customers(data_frame):
    """function to return top customers by quantity rented"""
    data_set = data_frame.groupby(['customer_id']).agg({'quantity_rented': 'sum'})
    data_set = data_set.nlargest(5, 'quantity_rented')
    return data_set


def top_ten_customers(data_frame):
    """function to return top top ten customers by total payments"""
    data_frame.loc[:, 'total_payment'] = (data_frame['unit_rental_price']
                                          * data_frame['quantity_rented']
                                          * data_frame['rental_period_months'])
    data_set = data_frame.groupby(['customer_id']).agg({'total_payment': 'sum'})
    data_set = data_set.nlargest(10, 'total_payment')
    return data_set


def bottom_twenty_customers(data_frame):
    """function to return bottom twenty customers by total payment"""
    data_frame.loc[:, 'total_payment'] = (data_frame['unit_rental_price']
                                          * data_frame['quantity_rented']
                                          * data_frame['rental_period_months'])
    data_set = data_frame.groupby(['customer_id']).agg({'total_payment': 'sum'})
    data_set = data_set.nsmallest(20, 'total_payment')
    return data_set


def write_file(file):
    """function to write csv file output"""
    file.to_csv('data_set.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    DATA = create_data()
    TOP_PRODUCTS = top_products(DATA)
    print("Top 10 products by Monthly Payments\n", TOP_PRODUCTS)
    TOP_FIVE_CUSTOMERS = top_five_customers(DATA)
    print("\n Top 5 customers by Quantity Rented\n", TOP_FIVE_CUSTOMERS)
    TOP_TEN_CUSTOMERS = top_ten_customers(DATA)
    print("\n Top 10 customers by Total Monthly Payments\n", TOP_TEN_CUSTOMERS)
    BOTTOM_TWENTY_CUSTOMERS = bottom_twenty_customers(DATA)
    print("\n Bottom 20 customers by Total Monthly Payments\n", BOTTOM_TWENTY_CUSTOMERS)
    write_file(DATA)
