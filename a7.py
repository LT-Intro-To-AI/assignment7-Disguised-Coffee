#interpreter command -->$ python 3
import time


from neural import NeuralNet

def run_test_xor(hidden_nodes):
    
    print("Running the XOR data Neural Net")
    xor_data = [
        ([0,0],[0]),
        ([0,1],[1]),
        ([1,0],[1]),
        ([1,1],[0]),]
    
    orn = NeuralNet(2,hidden_nodes,1)
    old_t = time.time()
    orn.train(xor_data)
    new_t = time.time()

    print(f"\nTime to train: {(new_t-old_t)}\n")
    
    print(orn.test_with_expected(xor_data))

def run_test_or():
    print("Running the OR data Neural Net")
    or_data = [
        ([0,0],[0]),
        ([0,1],[1]),
        ([1,0],[1]),
        ([1,1],[1]),]

    orn = NeuralNet(2,50,1)
    old_t = time.time()
    orn.train(or_data)
    new_t = time.time()

    print(f"\nTime to train: {(new_t-old_t)}\n")

    print(orn.test_with_expected(or_data))

def run_test_political():
    print("Lets not get political please.")

    actual_voter_opinion = [
        ([0.9,0.6,0.8,0.3,0.1],[1]),
        ([0.8,0.8,0.4,0.6,0.4],[1]),
        ([0.7,0.2,0.4,0.6,0.3],[1]),
        ([0.5,0.5,0.8,0.4,0.8],[0]),
        ([0.3,0.1,0.6,0.8,0.8],[0]),
        ([0.6,0.3,0.4,0.3,0.6],[0]),
    ]


    
    von = NeuralNet(5,50,1)
    old_t = time.time()
    von.train(actual_voter_opinion)
    new_t = time.time()

    print(f"\nTime to train: {(new_t-old_t)}\n")
    # print(von.test_with_expected(actual_voter_opinion))

    prediction_voter_opinion = [
        ([1.0,1.0,1.0,0.1,0.1],[]),
        ([0.5,0.2,0.1,0.7,0.7],[]),
        ([0.8,0.3,0.3,0.3,0.8],[]),
        ([0.8,0.3,0.3,0.8,0.3],[]),
        ([0.9,0.8,0.8,0.3,0.6],[]),
    ]
    # print(von.test_with_expected(actual_voter_opinion))
    print("Neural Network w/ 50 nodes")
    for tuple_pair in von.test_with_expected(prediction_voter_opinion):
        print(tuple_pair)

if __name__ == "__main__":
    run_test_political()
    # run_test_xor(1)
    # for i in range(10):
    #     run_test_political()