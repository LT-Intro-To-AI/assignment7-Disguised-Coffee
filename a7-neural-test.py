import time
from neural import NeuralNet
from typing import List, Tuple, Any
import pandas as pd
import numpy as np

"""
Import data into a list.

Specify the columns to put into data set
Specify the column as result.
"""

# target File
DATA_FILE = "testing_data/"
TARGET_FILE_NAME = "imports-85.data"

TARGET_DATA = DATA_FILE + TARGET_FILE_NAME


COLUMN_NAMES = "symboling,normalized-losses,make,fuel-type,aspiration,num-of-doors,body-style,drive-wheels,engine-location,wheel-base,length,width,height,curb-weight,engine-type,num-of-cylinders,engine-size,fuel-system,bore,stroke,compression-ratio,horsepower,peak-rpm,city-mpg,highway-mpg,price"


def get_data(cols: list) -> List:
    # iloc --> int index 
    # loc --> label indexing

    # names --> heading of CSV file
    
    # BTW, formulate values into 0-1
    # 3
    tor : list = []
    for i in range(1,205):
        pair = []
        for word in cols:
            # if word in COLUMN_VALUES:
            #     pair.append(convert_strings_to_values(file.iloc[i].at[word]),)
            pair.append(file.iloc[i].at[word])
        test_case = [file.iloc[i].at[file.columns[8]]]
        tuple_test : tuple = (pair,test_case)
        tor.append(tuple_test)
    return tor

def find_all_string_values():
    #Find all values within column and assign to a string
    pass


def format_to_neural(value, possible_values : list):
    # Divide value or value position by total amount of possible values.
    # ie: for symboling, -3 is 1 of 6 possible values.
    # so convert -3 to 1 and then divide by 6.
    
    #Value is a int
    if type(value) == int:
        # do int thing
        # min and max value must be in a tuple.
        if type(possible_values[0]) != tuple:
            print(f"ERROR @ format_to_neural: Possible value for number is not in a tuple!")
            exit
        else:
            difference = abs(possible_values[0][1] - possible_values[0][1])
            new_number = abs(value) - possible_values[0][1] # formatting
            return (new_number / difference)
    
    #Value is a string
    elif type(value) == str:
        # do string thing
        length = len(possible_values)
        index = possible_values.index(value)
        return (index/length)
    else:
        print(f"ERROR @ format_to_neural: can't process value \"{value}\"")
        exit

def format_from_neural(value, possible_values : list):
    # Find nearest value, based on possible values.
    # so convert -3 to 1 and then divide by 6.
    if type(possible_values[0]) == int:
        # do int thing
        # min and max value must be in a tuple.
        if type(possible_values[0]) != tuple:
            print(f"ERROR @ format_to_neural: Possible value for number is not in a tuple!")
            exit
        else:
            difference = abs(possible_values[0][1] - possible_values[0][1])
            new_number = abs(value) - possible_values[0][1] # formatting
            return (new_number / difference)
    
    #Value is a string
    elif type(possible_values[0]) == str:
        # do string thing
        length = len(possible_values)
        index = value * length
        return possible_values[closest_number((index/length), convert_strings_to_values(possible_values))]
    else:
        print(f"ERROR @ format_to_neural: can't process value \"{value}\"")
        exit

def closest_number(value, possible_values : list):
    # I bailed out on this. (help)
    arr = np.asarray(possible_values)

    i = (np.abs(arr - value)).argmin()

    return arr[i]

def convert_strings_to_values(string_list: list) -> list:
    # Convert the string to a list of values.
    index = 0
    tor = []
    for value in string_list:
        tor.append(index)
        index+=1
    return tor

def convert_value_to_string(int_list: list, string_list: list) -> list:
    # Convert the ints to a list of values
    tor = []
    for number in int_list:
        tor.append(string_list[number])
    return tor

def run_test(hidden_nodes : int, data : List) -> None:
    net = NeuralNet(2,hidden_nodes,1)
    old_t = time.time()
    net.train(data)
    new_t = time.time()

if __name__ == "__main__":
    file = pd.read_csv(TARGET_DATA, names=COLUMN_NAMES.split(","))
    # parse_data = get_data(file.columns[0:4])
    print(file["fuel-type"].shape)