import unittest

from lesson01extra_matthewdenko import *

class TestTuple(unittest.TestCase):

    def test_create_tuple(self):

        assert create_tuple() == [('Chair', 'A', 10),('Bed', 'A', 12),
            ('Chair', 'S', 6),
            ('Table', 'A', 10),
            ('Sofa', 'S', 20),
            ('Chair', 'S', 30),
            ('Bed', 'A', 100),
            ('Table', 'S', 15),
            ('Sofa', 'A', 15),
            ('Lamp', 'S', 20),
            ('Lamp', 'A', 45),
            ('Stereo', 'A', 1000),
            ('Table', 'A', 1000),
            ('Bed', 'S', 50),
            ('Lamp', 'S', 100),
            ('Bed', 'A', 100),
            ('Desk', 'A', 500),
            ('Desk', 'S', 100),
            ('Stereo', 'S', 100),
            ('Lamp', 'A', 50),
            ('Sofa', 'A', 10),
            ('Table', 'A', 8),
            ('Bed', 'A', 100),
            ('Table', 'S', 15),
            ('Sofa', 'A', 15),
            ('Lamp', 'S', 20),
            ('Lamp', 'A', 45),
            ('Stereo', 'A', 1000),
            ('Table', 'A', 1000),
            ('Bed', 'S', 50),
            ('Lamp', 'S', 100)]

    def test_tuple_magic(self):

        assert tuple_magic(create_tuple()) == [('Bed', 212), ('Chair', -26),
            ('Desk', 400), ('Lamp', -100), ('Sofa', 20), ('Stereo', 1900),
            ('Table', 1988)]


#    def test_():
#      tuple_2 = tuple_magic(input_tuple)
#      assert tuple_2 == [('Bed', 12),('Chair', 4)]


if __name__ == '__main__':
    unittest.main()
