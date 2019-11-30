import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    
    softmax_proba = []
    
    for i in range(len(L)):
        probability = np.exp(L[i]) / np.sum(np.exp(L))
        softmax_proba.append(probability)
        
    return softmax_proba
    
