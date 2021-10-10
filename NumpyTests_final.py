

#added comment testingbranch
# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from test_cases import *
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import pickle
import math


import time

start = time.time()
print("hello")


HE = Pyfhel()           

np.random.seed(5) # set a seed so that the results are consistent

X, Y = load_planar_dataset()


plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);


E=math.e


shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # training set size


print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);


# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# In[67]:


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
   
    return (n_x, n_h, n_y)


X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[70]:

n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


#sigmoid function calculation for every encrypted item in the numpy array
def sigmoid_test(array):
    temp=0
    for item in array:
        temp2=0
        for item2 in item:
            
            array[temp][temp2]= sigmoid_homomorphic(item2)
            temp2=temp2+1
        temp=temp+1

    return array


#tanh function calculation for every encrypted item in the numpy array
def tanh_test(arr):
  
    temp=0
    for item in arr:
        temp2=0
        for item2 in item:
            
            arr[temp][temp2]= tanh_homomorphic(item2)
            temp2=temp2+1
        temp=temp+1

    
    return arr

# function to encrypt all items from a numpy array 
def encrypt2darray(array1):

        arr_gen1 = np.empty(shape=(array1.shape[0],array1.shape[1]),dtype=PyCtxt)
        
        temp=0
        for item in array1:
            temp2=0
            for item2 in item:
               
                arr_gen1[temp][temp2] = HE.encryptFrac(item2)
                temp2=temp2+1
            temp=temp+1

        return arr_gen1

def decrypt_for_test(xtest):
    return_arr = np.empty(shape=(xtest.shape[0],xtest.shape[1]),dtype=PyCtxt)
    temp=0
    for item in xtest:
        temp2=0
        for item2 in item:
            
            return_arr[temp][temp2]= HE.decryptFrac(item2)
            temp2=temp2+1
        temp=temp+1
    return return_arr

# function to compute the approximation of the exponent of natural number e based on the MacLaurin series
# iterations are limited to 8 in order for our results to be decrypted accurately
def exponentialchronfhe(x):

    relinKeySize=50
    HE.relinKeyGen(bitCount=15, size=relinKeySize)

    x = ~ x
    sum=HE.encryptFrac(0)
    for i in range(8):
        if i ==0 :
            sum=HE.encryptFrac(1)
        else:
            #const=HE.encryptFrac(1/math.factorial(i))
            #const=~const
            x = ~ x
            sum=~sum
            
            const=1/math.factorial(i)
            
            const_power=(x**i)
            const_power=~const_power
            sum=sum+const_power*const
            print(sum,i,HE.decryptFrac(sum),const,HE.decryptFrac(x**i))
    return sum


#sigmoid formula computation using the exponential approximation function
	
def sigmoid_homomorphic(X):
    #we decrypt the divider and divisor in order to compute the result (we shall implement in a later version an approach to ciphertext w ciphertext division)
    return HE.encryptFrac((HE.decryptFrac(exponentialchronfhe(X))/(HE.decryptFrac(exponentialchronfhe(X)+HE.encryptFrac(1)))))

#tanh formula computation using the exponential approximation function

def tanh_homomorphic(X):
    relinKeySize=50
    HE.relinKeyGen(bitCount=15, size=relinKeySize)
    item=HE.encryptFrac(2)
    item=~item
    
    item2=item*X
    
    ar=exponentialchronfhe(item2)-HE.encryptFrac(1)
    par=exponentialchronfhe(item2)+HE.encryptFrac(1)
    #we decrypt the divider and divisor in order to compute the result (we shall implement in a later version an approach to ciphertext w ciphertext division)
    return HE.encryptFrac(HE.decryptFrac(ar)/HE.decryptFrac(par))

#forward propapagion function

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    #set the parameters for the context generation
    # p = plaintext modulus
    # m= coefficient modulus
    #flagbatching = option to use batching
    HE.contextGen(p=131072,m=8192,flagBatching=False,fracDigits=75,intDigits=75)  # Generating context. The value of p is important.
                            #  There are many configurable parameters on this step
                            #  More info in Demo_ContextParameters.py, and
                            #  in the docs of the function (link to docs in README)
    HE.keyGen() 
    
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
	
    W1_encrypted=encrypt2darray(W1)
    b1_encrypted=encrypt2darray(b1)
    W2_encrypted=encrypt2darray(W2)
    b2_encrypted=encrypt2darray(b2)
    X_encrypted=encrypt2darray(X)
   
   
    Z1 = np.dot(W1, X) + b1
    Z1_encrypted=np.dot(W1_encrypted,X_encrypted)+b1_encrypted
    print("Z1 original",Z1)
    print("Z1 after homomorphic",decrypt_for_test(Z1_encrypted))
  

    A1 = np.tanh(Z1)
    A1_encrypted=tanh_test(encrypt2darray(Z1))
    print("A1 original",A1)
    print("A1 after homomorphic encryption",decrypt_for_test(A1_encrypted))
   
    
    Z2 = np.dot(W2, A1) + b2
    Z2_encrypted=np.dot(W2_encrypted, A1_encrypted) + b2_encrypted
    print("Z2 original",Z2)
    print("Z2 after homomorphic encryption",decrypt_for_test(Z2_encrypted))
   
    
    A2 = sigmoid(Z2)
    A2_encrypted = sigmoid_test(Z2_encrypted)
    print("sigmoid original",A2)
    print("sigmoid after homomorphic encryption",decrypt_for_test(A2_encrypted))
    

    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# In[79]:


X_assess, parameters = forward_propagation_test_case()

A2, cache = forward_propagation(X_assess, parameters)



#cost computation function

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example
    
   
    W1 = parameters['W1']
    W2 = parameters['W2']
	
    X_encrypted=encrypt2darray(X)
    Y_encrypted=encrypt2darray(Y)
    W1_encrypted=encrypt2darray(W1)
    W2_encrypted=encrypt2darray(W2)
    A2_encrypted=encrypt2darray(A2)
  
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1-A2))
    temp=Y_encrypted
    temp=temp*HE.encryptFrac(-1)
    temp=temp+HE.encryptFrac(1)
    #np.multiply with ciphertext-object numpy array works similar to numpy.dot operation with ciphertexts
    logprobs_encrypted = np.multiply(encrypt2darray(np.log(A2)), Y_encrypted) + np.multiply(temp, encrypt2darray(np.log(1-A2)))
    print("logprobs original",logprobs)
    print("logprobs after homomorphic",decrypt_for_test(logprobs_encrypted),logprobs_encrypted)
   
    
    cost = - np.sum(logprobs) / m
    cost_encrypted = ( np.sum(logprobs_encrypted) *HE.encryptFrac(-1))*HE.encryptFrac(1/m)
    print("cost original", cost)
    print("cost after homomorphic", HE.decryptFrac(cost_encrypted))

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    #cost_encrypted=np.squeeze(cost_encrypted)
    print("cost original", cost)
    print("cost after homomorphic", HE.decryptFrac(cost_encrypted))
    assert(isinstance(cost, float))
    
    return cost



A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


#backward propagation function

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
	#we set the context parameters for the encryption on the client side
    #we evaluate it here for testing purposes
    HE.contextGen(p=131072,m=16384,flagBatching=False,fracDigits=75,intDigits=75)  # Generating context. The value of p is important.
                            #  There are many configurable parameters on this step
                            #  More info in Demo_ContextParameters.py, and
                            #  in the docs of the function (link to docs in README)
    HE.keyGen() 

    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
  
    W1 = parameters['W1']
    W2 = parameters['W2']

        

    A1 = cache['A1']
    A2 = cache['A2']
   
    W1_encrypted=encrypt2darray(W1)
    W2_encrypted=encrypt2darray(W2)
    A1_encrypted=encrypt2darray(A1)
    A2_encrypted=encrypt2darray(A2)
	
    Y_encrypted=encrypt2darray(Y)
    X_encrypted=encrypt2darray(X)
   
    dZ2= A2 - Y
    dZ2_encrypted=A2_encrypted-Y_encrypted
    fraction_encrypted=HE.encryptFrac(1 / m)
	
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    dW2_encrypted= np.dot(dZ2_encrypted, A1_encrypted.T)
    dW2_encrypted=dW2_encrypted*HE.encryptFrac(1 / m)
    print("dw2 original",dW2)
    print("dw2 after homomorphic",decrypt_for_test(dW2_encrypted),dW2_encrypted)
  
    
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    db2_encrypted= np.sum(dZ2_encrypted, axis=1, keepdims=True)
    db2_encrypted=db2_encrypted*HE.encryptFrac(1 / m)
    print("db2 original",db2)
    print("db2 after homomorphic",decrypt_for_test(db2_encrypted),db2_encrypted)
    
  
   
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    temp=A1_encrypted**2
    temp=temp*HE.encryptFrac(-1)
    temp=temp+HE.encryptFrac(1)
   
   
    dZ1_encrypted = np.multiply(np.dot(W2_encrypted.T, dZ2_encrypted),temp)
    
    print("dZ1 original",dZ1)
    print("dZ1 after homomorphic",decrypt_for_test(dZ1_encrypted),dZ1_encrypted)
   
    

    dW1 = (1 / m) * np.dot(dZ1, X.T)
    dW1_encrypted = np.dot(dZ1_encrypted, X_encrypted.T)
  
    dW1_encrypted=dW1_encrypted*HE.encryptFrac(1 / m)
    print("np.dot(dZ1, X.T) original",np.dot(dZ1, X.T))
    print("np.dot(dZ1, X.T) after homomorphic",decrypt_for_test(np.dot(dZ1_encrypted, X_encrypted.T)))
    print("dW1 original",dW1)
    print("dW1 after homomorphic",decrypt_for_test(dW1_encrypted),dW1_encrypted)

    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    db1_encrypted =  np.sum(dZ1_encrypted, axis=1, keepdims=True)
    db1_encrypted=db1_encrypted*HE.encryptFrac(1 / m)
    print("db1 original",db1)
    print("db1 after homomorphic",decrypt_for_test(db1_encrypted))
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads



parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


end = time.time()
print(end - start)









