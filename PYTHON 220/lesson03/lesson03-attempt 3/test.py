import unittest

from basic_operations import *

class TestDatabase(unittest.TestCase):

    def test_create_table(self):
        DB = SqliteDatabase('people.db')
        DB.connect()
        create_tables(DB)
        assert DimCustomer == DimCustomer

    def test_add_customer(self):
        DB = SqliteDatabase('people.db')
        DB.connect()
        create_tables(DB)
        RICHIE = add_customer(1, 'RICHIE', 'rich', '1010 Rich Dr.',
                          '777-666-5555', 'rich@rich.net', 'Active', 1000)
        DONALD = add_customer(2, 'DONALD', 'duck', 'The Swamp',
                          '777-666-5553', 'donald@duck.net', 'Active', 11000)
        BILL = add_customer(3, 'BILL', 'gates', 'Medina',
                        '248-666-5553', 'bill@gates.net', 'Inactive', 50)
        """testing adding a new customer"""
        assert RICHIE.name == 'RICHIE'
        assert DONALD.phone_number == '777-666-5553'
        assert BILL.email_address == 'bill@gates.net'

    def test_search_customer(self):
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
        assert SEARCH_DICTIONARY == SEARCH_DICTIONARY


    def test_list_active_customers(self):
        DB = SqliteDatabase('people.db')
        DB.connect()
        create_tables(DB)
        RICHIE = add_customer(1, 'RICHIE', 'rich', '1010 Rich Dr.',
                          '777-666-5555', 'rich@rich.net', 'Active', 1000)
        DONALD = add_customer(2, 'DONALD', 'duck', 'The Swamp',
                          '777-666-5553', 'donald@duck.net', 'Active', 11000)
        BILL = add_customer(3, 'BILL', 'gates', 'Medina',
                        '248-666-5553', 'bill@gates.net', 'Inactive', 50)
        ACTIVE_CUSTOMERS = list_active_customers()
        assert ACTIVE_CUSTOMERS == ACTIVE_CUSTOMERS

    def test_dimcustomer(self):
        assert DimCustomer == DimCustomer
        assert DimCustomer.name == DimCustomer.name
        assert DimCustomer.last_name == DimCustomer.last_name

    def test_delete_customer(self):
        DB = SqliteDatabase('people.db')
        DB.connect()
        create_tables(DB)
        RICHIE = add_customer(1, 'RICHIE', 'rich', '1010 Rich Dr.',
                          '777-666-5555', 'rich@rich.net', 'Active', 1000)
        DONALD = add_customer(2, 'DONALD', 'duck', 'The Swamp',
                          '777-666-5553', 'donald@duck.net', 'Active', 11000)
        BILL = add_customer(3, 'BILL', 'gates', 'Medina',
                        '248-666-5553', 'bill@gates.net', 'Inactive', 50)
        DB = SqliteDatabase('people.db')
        delete = delete_customer(1)
        assert delete == delete

    def test_update_credit_limit(self):
        DB = SqliteDatabase('people.db')
        DB.connect()
        create_tables(DB)
        RICHIE = add_customer(1, 'RICHIE', 'rich', '1010 Rich Dr.',
                          '777-666-5555', 'rich@rich.net', 'Active', 1000)
        DONALD = add_customer(2, 'DONALD', 'duck', 'The Swamp',
                          '777-666-5553', 'donald@duck.net', 'Active', 11000)
        BILL = add_customer(3, 'BILL', 'gates', 'Medina',
                        '248-666-5553', 'bill@gates.net', 'Inactive', 50)
        DB = SqliteDatabase('people.db')
        update = update_customer_credit(1,100)
        assert update == update

if __name__ == '__main__':
    unittest.main()
