#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Importing os packages and modules from peewee"""
from peewee import SqliteDatabase, Model, IntegerField, CharField, DecimalField

DB = SqliteDatabase('people.db')

class DimCustomer(Model):
    """Class for database model"""
    customer_id = IntegerField(primary_key=True)
    name = CharField()
    last_name = CharField()
    home_address = CharField()
    phone_number = CharField()
    email_address = CharField()
    credit_limit = DecimalField()
    status = CharField()
    class Meta: # pylint: disable=R0903
        """Meta class for database"""
        database = DB


def create_tables(database):
    """Function to create the table in the database"""
    database.create_tables([DimCustomer])


def add_customer(customer_id, name, last_name, home_address, phone_number, # pylint: disable=R0913
                 email_address, status, credit_limit):
    """Function to add a new customer to the database"""
    new_customer = DimCustomer(customer_id=customer_id,
                               name=name,
                               last_name=last_name,
                               home_address=home_address,
                               phone_number=phone_number,
                               email_address=email_address,
                               status=status,
                               credit_limit=credit_limit)
    new_customer.save()
    return new_customer


def search_customer(customer_id):
    """Function to search for a customer in the database by customer id"""
    query = (DimCustomer.select().where(DimCustomer.customer_id
                                        == customer_id))
    output_dict = dict()
    for customer in query:
        output_dict.update({'name': customer.name,
                            'last_name': customer.last_name,
                            'email_address': customer.email_address,
                            'phone_number': customer.phone_number})
    print(output_dict)
    return dict(output_dict)


def update_customer_credit(customer_id, credit_limit):
    """Function to update credit limit of customer"""
    try:
        query = (DimCustomer.select().where(DimCustomer.customer_id
                                            == customer_id))
        for customer in query:
            customer.credit_limit = credit_limit
            customer.save()
    except ValueError:
        print("enter a valid credit_limit")


def delete_customer(customer_id):
    """Function to remove a customer from the database"""
    query = (DimCustomer.select().where(DimCustomer.customer_id ==
                                        customer_id))
    for customer in query:
        customer.delete_instance()


def list_active_customers():
    """Function to return all active customers"""
    query = (DimCustomer.select().where(DimCustomer.status == 'Active'))
    output_dict = dict()
    for customer in query:
        output_dict.update({'id': customer.customer_id,
                            'name': customer.name,
                            'last_name': customer.last_name,
                            'status': customer.status})
    return dict(output_dict)


if __name__ == "__main__":
    DB = SqliteDatabase('people.db')
    DB.connect()
    create_tables(DB)
    RICHIE = add_customer(1, 'RICHIE', 'rich', '1010 Rich Dr.',
                          '777-666-5555', 'rich@rich.net', 'Active', 1000)
    DONALD = add_customer(2, 'DONALD', 'duck', 'The Swamp',
                          '777-666-5553', 'donald@duck.net', 'Active', 11000)
    BILL = add_customer(3, 'BILL', 'gates', 'Medina',
                        '248-666-5553', 'bill@gates.net', 'Inactive', 50)
    SEARCH_DICTIONARY = search_customer(RICHIE.customer_id)
    update_customer_credit(2, 75)
    delete_customer(1)
    ACTIVE_CUSTOMERS = list_active_customers()
    DB.close()
