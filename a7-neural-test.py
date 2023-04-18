import time
from neural import NeuralNet
from typing import List, Tuple, Any
import pandas as pd

"""
Import data into a list.

Specify the columns to put into data set
Specify the column as result.
"""

# target File
DATA_FILE = "testing_data/"
TARGET_FILE_NAME = "imports-85.data"

TARGET_DATA = DATA_FILE + TARGET_FILE_NAME

def get_data() -> List:
    # iloc --> int index 
    # loc --> label indexing

    # names --> heading of CSV file
    names = "symboling,normalized-losses,make,fuel-type,aspiration,num-of-doors,body-style,drive-wheels,engine-location,wheel-base,length,width,height,curb-weight,engine-type,num-of-cylinders,engine-size,fuel-system,bore,stroke,compression-ratio,horsepower,peak-rpm,city-mpg,highway-mpg,price"
    file = pd.read_csv(TARGET_DATA, names=names.split(","))
    test = 0
    cols = file.columns[0:4]
    
    # BTW, formulate values into 0-1
    # 3
    for i in range(1,205):
        pair = []
        for word in cols:
            pair.append(file.iloc[i].at[word])
        test_case = [file.iloc[i].at[file.columns[8]]]
        tuple_test = (pair,test_case)
        print(tuple_test)

def run_test(hidden_nodes : int, data : List) -> None:
    net = NeuralNet(2,hidden_nodes,1)
    old_t = time.time()
    net.train(data)
    new_t = time.time()

if __name__ == "__main__":
    get_data()