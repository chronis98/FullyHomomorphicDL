#!/usr/bin/env python
# coding: utf-8

# In[61]:


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
#get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1) # set a seed so that the results are consistent


# In[62]:


X, Y = load_planar_dataset()


# In[63]:


plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);


# In[64]:


### START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # training set size
### END CODE HERE ###

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


# In[65]:


clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);


# In[66]:


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
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    ### END CODE HERE ###
    return (n_x, n_h, n_y)


# In[68]:


X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


# In[69]:

 

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
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    ### END CODE HERE ###
    
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


# In[78]:

def fhe_computation(X,Y):
    HE = Pyfhel()           # Creating empty Pyfhel object
    HE.contextGen(p=65537)  # Generating context. The value of p is important.
                            #  There are many configurable parameters on this step
                            #  More info in Demo_ContextParameters.py, and
                            #  in the docs of the function (link to docs in README)
    HE.keyGen()             # Key Generation.
    print(HE)
    print("Xronakos",X)
    print("Xronakos2",Y)
    print("2. Encrypting two arrays of integers.")
    print("    For this, you need to create empty arrays in numpy and assign them the cyphertexts")
    #array1 = np.array(X.flatten())
    Y_n=Y.flatten()
    if isinstance(X, list):
        array1 = np.array(X)
    else:
        array1=np.full((Y.shape[0],Y.shape[1]), X)

    #if isinstance(Y, list):
        
    #array1 = np.array(Y_n)
    #else:
        #array2=np.full((X.shape[0],X.shape[1]), Y)
    print("Xronis testarei",Y_n)
    #array2 = np.array([-2., 4., -6., 8.,2.,-6., 8.,2.,-6., 8.,2.,3.])
    array1=array1.flatten()
    array2 = np.array(Y.flatten())
    arithmos=len(array2)
    
    arr_gen1 = np.empty(len(array1),dtype=PyCtxt)
    arr_gen2 = np.empty(len(array2),dtype=PyCtxt)

    # Encrypting! This can be parallelized!
    for i in np.arange(len(array1)):
        arr_gen1[i] = HE.encryptFrac(array1[i])
        arr_gen2[i] = HE.encryptFrac(array2[i])

    print("    array1: ",array1,'-> ctxt1 ', type(arr_gen1), ', dtype:', arr_gen1.dtype)
    print("    array2: ",array2,'-> ctxt2 ', type(arr_gen2), ', dtype:', arr_gen2.dtype)

    print("3. Sending the data and the pyfhel object to operate.")
    # NETWORK! Pickling
    # Pickling to send it over the network. Each PyCtxt can be pickled individually.
    # arr1_pk and arr2_pk are strings that could be sent over the network or saved.
    import pickle
    arr1_pk = pickle.dumps(arr_gen1)
    arr2_pk = pickle.dumps(arr_gen2)
    pyfh_pk = pickle.dumps(HE)

    # Here you would send the pickled strings over the network

    # Retrieve data
    arr_ctxt1 = pickle.loads(arr1_pk)
    arr_ctxt2 = pickle.loads(arr2_pk)
    HE2 = pickle.loads(pyfh_pk)

    # Reassign Pyfhel instance. Pickling doesn't keep the pyfhel instance
    for ctxt in arr_ctxt1:
        ctxt._pyfhel = HE2

    print("4. Vectorized operations with encrypted arrays of PyCtxt")
    ctxtSum = arr_ctxt1 + arr_ctxt2         # `ctxt1 += ctxt2` for quicker inplace operation
    ctxtSub = arr_ctxt1 - arr_ctxt2         #   `ctxt1 -= ctxt2` for quicker inplace operation
    ctxtMul = arr_ctxt1 * arr_ctxt2         # `ctxt1 *= ctxt2` for quicker inplace operation

    # Here you could send back the resulting ciphertexts pickling them.


    print("4. Decrypting results:")
    resSum = [HE.decryptFrac(ctxtSum[i]) for i in np.arange(len(ctxtSum))]
    resSub = [HE.decryptFrac(ctxtSub[i]) for i in np.arange(len(ctxtSub))] 
    resMul = [HE.decryptFrac(ctxtMul[i]) for i in np.arange(len(ctxtMul))]
    print("     addition:       decrypt(ctxt1 + ctxt2) =  ", resSum)
    print("     substraction:   decrypt(ctxt1 - ctxt2) =  ", resSub)
    print("     multiplication: decrypt(ctxt1 + ctxt2) =  ", resMul)
    return resMul
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    

    ### START CODE HERE ### (≈ 2 lines of code)
   
    #apotelesma=fhe_computation(np.dot(W1, X),b1)
   # print("Z1 test",apotelesma)
    
    Z1 = np.dot(W1, X) + b1
    print("Z1 real",Z1)
    A1 = np.tanh(Z1)
    #apotelesma2=fhe_computation(np.dot(W2, A1),b2)
    #print("Z2 test",apotelesma2)
    Z2 = np.dot(W2, A1) + b2
    print("Z2 real",Z2)
    A2 = sigmoid(Z2)
    ### END CODE HERE ###

    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# In[79]:


X_assess, parameters = forward_propagation_test_case()

A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))








# GRADED FUNCTION: compute_cost

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
    
    # Retrieve W1 and W2 from parameters
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters['W1']
    W2 = parameters['W2']
    ### END CODE HERE ###
    
    # Compute the cross-entropy cost
    #apotelesma3=fhe_computation(np.multiply(np.log(A2), Y),np.multiply((1 - Y), np.log(1 - A2)))
    #print("logprobs test",apotelesma3)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    print("logprobs real",logprobs)
    #logprobs = resMul+ np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost


# In[74]:



A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# In[75]:


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
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters['W1']
    W2 = parameters['W2']
    ### END CODE HERE ###
        
    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache['A1']
    A2 = cache['A2']
    ### END CODE HERE ###
    
    # Backward propagation: calculdW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2= A2 - Y
   
    result=fhe_computation((1 / m), np.dot(dZ2, A1.T))
    print("dw2 test",result)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    print("dw2 real",dW2)
    result2=fhe_computation((1 / m), np.sum(dZ2, axis=1, keepdims=True))
    print("db2 test",result2)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    print("dw2 real",db2)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    result3=fhe_computation((1 / m), np.dot(dZ1, X.T))
    print("dW1 test",result3)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    print("dW1 real",dW1)
    result4=fhe_computation((1 / m), np.sum(dZ2, axis=1, keepdims=True))
    print("db1 test",result4)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    print("db1 real",db1)
    ### END CODE HERE ###
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# In[76]:


parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


# In[ ]:










