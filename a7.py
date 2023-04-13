#interpreter command -->$ python 3

from neural import NeuralNet

xor_data = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[1]),]

or_data = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[0]),]

orn = NeuralNet(2,3,1)
orn.train(xor_data)

print(orn.test_with_expected(xor_data))