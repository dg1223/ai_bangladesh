import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

def cross_entropy(Y, P):
    
    x1 = - np.sum( Y * np.log(P) )
    
    Y_hat = [ (1 - i) for i in Y]
    P_hat = [ (1 - j) for j in P]
    
    x2 = - np.sum( Y_hat * np.log(P_hat) )
    
    #print('Y data type = ', type(Y))
    #print('P data type = ', type(P))
    #print('Y_hat data type = ', type(Y_hat))
    #print('P_hat data type = ', type(P_hat))
    
    return x1 + x2
