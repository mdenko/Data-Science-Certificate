#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""importing functions"""
import functools
import pandas as pd

def create_invoice():
    """function to create_invoice"""
    invoice_file = pd.DataFrame(list())
    invoice_file.to_csv('invoice_file.csv')
    return invoice_file


def create_rental_items(invoice_file):
    """function to create_rental_items"""
    rental_items = pd.DataFrame()
    if not invoice_file.empty:
        rental_items.append(invoice_file)
    return rental_items


def add_furniture(invoice_file, customer_name, item_code, item_description, item_monthly_price):
    """function will create invoice file (if it doesn't exist) or append to new line if does"""
    customer_name_list = list()
    item_code_list = list()
    item_description_list = list()
    item_monthly_price_list = list()
    customer_name_list.append(customer_name)
    item_code_list.append(item_code)
    item_description_list.append(item_description)
    item_monthly_price_list.append(item_monthly_price)
    if not invoice_file.empty:
        data_frame = pd.DataFrame()
        data_frame['customer_name'] = customer_name_list
        data_frame['item_code'] = item_code_list
        data_frame['item_description'] = item_description_list
        data_frame['item_monthly_price'] = item_monthly_price_list
        invoice_file.append(data_frame)
    else:
        invoice_file = pd.DataFrame()
        invoice_file['customer_name'] = customer_name_list
        invoice_file['item_code'] = item_code_list
        invoice_file['item_description'] = item_description_list
        invoice_file['item_monthly_price'] = item_monthly_price_list
    return invoice_file


def return_single_customer(rental_items):
    """function to be called by single_customer"""
    return print(rental_items['customer_name'])


def single_customer(customer_name, invoice_file):
    """returns a function that takes one parameter, rental_items"""
    rental_items = pd.DataFrame()
    rental_items = invoice_file.loc[invoice_file['customer_name'] == customer_name]
    functools.partial(return_single_customer, rental_items)


if __name__ == "__main__":
    INVOICE_FILE = create_invoice()
    INVOICE_FILE = add_furniture(INVOICE_FILE, "Matt", "A346", "Baseball", 10.50)
    single_customer("Matt", INVOICE_FILE)
