# -*- coding: utf-8 -*-

"""Pandas and numpy"""
import pandas as pd
import numpy as np



def create_tuple():
    """function to create input tuple"""
    return [('Chair', 'A', 10),
            ('Bed', 'A', 12),
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

def tuple_magic(tuple_input):
    """function to return consolidated tuple"""
    data_frame = pd.DataFrame()
    type_2 = list()
    add_sell = list()
    amount = list()
    for i in range(0, len(tuple_input)):
        type_2.append(tuple_input[i][0])
        add_sell.append(tuple_input[i][1])
        amount.append(tuple_input[i][2])
        data_frame = pd.DataFrame()
        data_frame['Type'] = np.array(type_2)
        data_frame['Add_sell'] = np.array(add_sell)
        data_frame['Amount'] = np.array(amount)
    for i in range(0, len(tuple_input)):
        if data_frame['Add_sell'][i] == 'S':
            data_frame['Amount'][i] = data_frame['Amount'][i] * -1
        else:
            pass
    data_frame = data_frame.groupby('Type').sum()
    data_frame['Type'] = data_frame.index
    data_frame = data_frame[['Type', 'Amount']]
    tuple_input = [tuple(data_frame) for data_frame in data_frame.to_numpy()]
    return tuple_input


if __name__ == '__main__':
    INPUT_TUPLE = create_tuple()
    OUTPUT_TUPLE = tuple_magic(INPUT_TUPLE)
