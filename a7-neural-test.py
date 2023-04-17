import time
from neural import NeuralNet
from typing import List, Tuple, Any
import pandas as pd

# target File
DATA_FILE = "testing_data/"
TARGET_FILE_NAME = "imports-85.data"

TARGET_DATA = DATA_FILE + TARGET_FILE_NAME

def get_data() -> List:
    file = pd.read_csv(TARGET_DATA)
    print(file.to_string())


def run_test(hidden_nodes : int, data : List) -> None:
    
    net = NeuralNet(2,hidden_nodes,1)
    old_t = time.time()
    net.train(data)
    new_t = time.time()

if __name__ == "__main__":
    get_data()