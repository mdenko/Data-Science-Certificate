import unittest

from database import *
from pandas._testing import assert_frame_equal

class TestDatabase(unittest.TestCase):

    def test_import_data(self):
        directory_name = os.getcwd()
        customer_file = "customer_file.csv"
        product_file = "product_file.csv"
        rentals_file = "rentals_file.csv"
        products, customers, rentals, rental_customer = import_data(directory_name,product_file,customer_file,rentals_file)
        products_test = pd.read_csv(product_file)
        customers_test = pd.read_csv(customer_file)
        rentals_test = pd.read_csv(rentals_file)
        assert_frame_equal(products, products_test)
        assert_frame_equal(customers, customers_test)
        assert_frame_equal(rentals, rentals_test)
        print("import test passed")
        
if __name__ == '__main__':
    unittest.main()
