""" 
Softmax Regression (Multinomial Logistic, Maximum Entropy Classifier, or just Multi-class Logistic Regression) is a generalization of logistic regression that we can use for multi-class classification (under the assumption that the classes are mutually exclusive). In contrast, we use the (standard) Logistic Regression model in binary classification tasks. 

Other algorithms used:
    1. Gradient Descent Algorithm
    2. Back-Propagation Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt

# Initialiation of Weight and Bias - Array Sizes 
# Individual Elements of Weights and Bias are initialized to Zero
def initialization(dim1, dim2):
    w = np.zeros(shape=(dim1, dim2))
    b = np.zeros(shape=(10, 1))
    return w, b

''' Forward and Backward Propagations'''
def propagation(w, b, X, Y): 
    """
    Parameters:
        1. Weights
        2. Biases
        3. X Training Set
        4. True Labels - Y
    """
    m = X.shape[1]  # The Number of Rows

    # Forward Propogation
    A = softmax((np.dot(w.T,X)+b).T)
    cost = (-1/m)*np.sum(Y*np.log(A))

    # Backward Propogation
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)

    cost = np.squeeze(cost)
    gradients = {"dw": dw,
             "db": db}
    return gradients, cost

''' The optimization (weight updation) function '''
def Optimization(w, b, X, Y, num_iters, alpha, print_cost=False):
    
    """
    Parameters:
        1. Weights
        2. Biases
        3. X Training
        4. Y Training
        5. Number of Iterations
        6. Learning Rate
        7. Printing Error Cost
    """

    costs = []
    for i in range(num_iters):
        gradients, cost = propagation(w, b, X, Y)
        dw = gradients["dw"]
        db = gradients["db"]
        
        # Weight Updations
        w = w-alpha*dw
        b = b-alpha*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Printing the error cost per 100 training examples
        if print_cost and i % 100 == 0:
            print("Error cost after iteration %i is %f." % (i, cost))

    params = {"w": w,
              "b": b} # The weights and biases

    gradients = {"dw": dw,
             "db": db} # Change in weights and biases that is to be made in order to optimize the model

    return params, gradients, costs


def predict(w, b, X):
    
    # m = X.shape[1]
    # y_pred = np.zeros(shape=(1, m))
    # w = w.reshape(X.shape[0], 1)

    y_pred = np.argmax(softmax((np.dot(w.T, X) + b).T), axis=0)
    return y_pred

# Defining the Softmax Function
def softmax(z):
    z -= np.max(z)
    softmax = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return softmax


def model(X_train, Y_train, Y,X_test,Y_test, num_iters, alpha, print_cost):
    
    """
    Parameters:
    1. X_train
    2. Y_train
    3. Y - Actual labels
    4. X_test
    5. Y_test
    6. num_iterations - Number of Iterations in the Training Phase
    7. Learning Rate - Ideally 0.02 - 0.05
    8. Printing the Error Cost per Iteration. 
    """

    w, b = initialization(X_train.shape[0], Y_train.shape[0]) # Shape of X_train = Y_train = 8000
    parameters, gradients, costs = Optimization(w, b, X_train, Y_train, num_iters, alpha, print_cost)  

    w = parameters["w"]
    b = parameters["b"]

    y_prediction_train = predict(w, b, X_train)
    y_prediction_test = predict(w, b, X_test)
    
    print("Training Phase accuracy percentage: ", sum(y_prediction_train==Y)/(float(len(Y)))*100)
    print("Testing Phase accuracy percentage: ", sum(y_prediction_test==Y_test)/(float(len(Y_test)))*100)

    d = {"costs": costs,
         "Y_prediction_test": y_prediction_test,
         "Y_prediction_train": y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": alpha,
         "num_iterations": num_iters}

    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per 100s)')
    plt.title("Learning Rate =" + str(d["learning_rate"]))
    plt.plot()
    plt.show()

    return d
