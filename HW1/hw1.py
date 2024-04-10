###### Your ID ######
# ID1: 322462920
# ID2: 207827825
#####################

# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    # calculate the mean, min, and max of X and y
    mean_X = np.mean(X, axis=0)
    mean_y = np.mean(y, axis=0)
    min_X = np.min(X, axis=0)
    min_y = np.min(y, axis=0)
    max_X = np.max(X, axis=0)
    max_y = np.max(y, axis=0)

    # create arrays with the same shape as X and y
    mean_X_array = np.full(X.shape, mean_X)
    mean_y_array = np.full(y.shape, mean_y)
    min_X_array = np.full(X.shape, min_X)
    min_y_array = np.full(y.shape, min_y)
    max_X_array = np.full(X.shape, max_X)
    max_y_array = np.full(y.shape, max_y)

    # mean normalization
    X = (X - mean_X_array) / (max_X_array - min_X_array)
    y = (y - mean_y_array) / (max_y_array - min_y_array)
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    temp = np.full(X.shape[0], 1)  # create an array of ones
    X = np.column_stack((temp, X))  # add the array of ones as the first column
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0
    h_theta_X = np.sum(theta * X, axis=1)
    J = (1 / (2 * X.shape[0])) * np.sum(np.square(h_theta_X - y))
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m = X.shape[0]

    for i in range(num_iters):
        h_theta_X = np.dot(X, theta)
        theta = theta - alpha * (1 / m) * np.dot(X.T, (h_theta_X - y))
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []
    transpose_X = np.transpose(X)
    pinv_theta = np.dot(np.dot(np.linalg.inv(
        np.dot(transpose_X, X)), transpose_X), y)
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop
    the learning process once the improvement of the loss value is smaller
    than 1e-8. This function is very similar to the gradient descent
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m = X.shape[0]
    i = 0

    # stop the learning process once the improvement of the loss value is smaller than 1e-8 or we hit the maximum number of iterations
    while i < num_iters and (len(J_history) < 2 or J_history[-2] - J_history[-1] >= 1e-8):
        h_theta_X = np.dot(X, theta)
        theta = theta - alpha * (1 / m) * np.dot(X.T, (h_theta_X - y))
        J_history.append(compute_cost(X, y, theta))
        i += 1
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using
    the training dataset. maintain a python dictionary with alpha as the
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003,
              0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}

    # Theta is random according to the seed
    np.random.seed(42)
    theta = np.random.rand(X_train.shape[1])

    for alpha in alphas:
        alpha_dict[alpha] = compute_cost(X_val, y_val, efficient_gradient_descent(
            X_train, y_train, theta, alpha, iterations)[0])

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    for i in range(5):
        np.random.seed(42)
        theta = np.random.rand(len(selected_features) + 2)

        min_cost = float('inf')  # initialize min_cost to infinity
        best_feature = -1

        for j in range(X_train.shape[1]):
            if j not in selected_features:
                temp_features = selected_features.copy()
                temp_features.append(j)
                temp_X_train = apply_bias_trick(X_train[:, temp_features])
                temp_X_val = apply_bias_trick(X_val[:, temp_features])
                cost = compute_cost(temp_X_val, y_val, efficient_gradient_descent(
                    temp_X_train, y_train, theta, best_alpha, iterations)[0])
                if cost < min_cost:
                    # update the best feature and the minimum cost
                    min_cost = cost
                    best_feature = j
        selected_features.append(best_feature)
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()  # copy the input dataframe
    for i in range(df.shape[1]):
        for j in range(i, df.shape[1]):
            if i == j:
                name = df.columns[i] + '^2'
            else:
                name = df.columns[i] + '*' + df.columns[j]

            # create the new multiplied column
            column = df.iloc[:, i] * df.iloc[:, j]
            # add the new column to the dataframe
            df_poly = pd.concat((df_poly, column.rename(name)), axis=1)
    return df_poly
