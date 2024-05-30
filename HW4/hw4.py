import numpy as np
import pandas as pd


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    mean_x = 0.0
    mean_y = 0.0
    sigma_x_y = 0.0
    sigma_x_square = 0.0
    sigma_y_square = 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    sigma_x_y = np.sum((x - mean_x) * (y - mean_y))
    sigma_x_square = np.sum((x - mean_x) ** 2)
    sigma_y_square = np.sum((y - mean_y) ** 2)

    # check if both of the variables is homogeneous
    if (sigma_x_square == 0 and sigma_y_square == 0):
        return None
    # check if only one of the variables is homogeneous
    if (sigma_x_square == 0 or sigma_y_square == 0):
        return 0

    r = sigma_x_y / np.sqrt(sigma_x_square * sigma_y_square)

    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []

    # remove unnumeric columns
    X = X.select_dtypes(include=[np.number])

    # calculate the correlation between each feature and the target and sort the features by the correlation in absolote value
    correlations = X.apply(lambda x: pearson_correlation(x, y))
    correlations = correlations.abs().sort_values(ascending=False)

    # select the best n_features
    best_features = correlations.index[:n_features].tolist()

    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        # add bias to the data
        X = np.column_stack((np.ones(X.shape[0]), X))

        # init theta
        self.theta = np.random.random(X.shape[1])

        # init variables
        J_prev = 0
        J = 0
        i = 0

        # loop over the number of iterations
        while i < self.n_iter:
            # calculate the cost
            J = self._cost_function(X, y)
            # calculate the gradient
            grad = self._get_gradient(X, y)
            # update the theta
            self.theta = self.theta - self.eta * grad
            # save the cost
            self.Js.append(J)
            # save the theta
            self.thetas.append(self.theta)
            # check for convergence

            if np.abs(J - J_prev) < self.eps:
                break
            J_prev = J
            i += 1

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        m = X.shape[0]
        h = self._sigmoid(X.dot(self.theta))
        J = -1/m * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))
        return J

    def _get_gradient(self, X, y):
        m = X.shape[0]
        h = self._sigmoid(X.dot(self.theta))
        # grad = 1/m * X.T.dot(h-y)
        grad = X.T.dot(h-y)

        return grad

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        # add bias to the data
        X = np.column_stack((np.ones(X.shape[0]), X))
        # calculate the probability
        probs = self._sigmoid(X.dot(np.asarray(self.theta)))
        # classify the data
        preds = np.where(probs >= 0.5, 1, 0)

        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    # shuffle the data
    idx = np.random.permutation(X.shape[0])
    X_copy = X[idx]
    y_copy = y[idx]

    # split the data to folds
    X_folds = np.array_split(X_copy, folds)
    y_folds = np.array_split(y_copy, folds)

    # loop over the folds
    accuracies = []
    for i in range(folds):
        # get the train and test data
        X_train = np.concatenate([X_folds[j] for j in range(folds) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(folds) if j != i])
        X_test = X_folds[i]
        y_test = y_folds[i]

        # train the model
        algo.fit(X_train, y_train)

        # predict the test data
        preds = algo.predict(X_test)

        # calculate the accuracy
        accuracy = np.mean(preds == y_test)
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)

    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """

    p = None
    p = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((data.reshape(-1, 1) - mu) / sigma) ** 2)  # nopep8
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    # def init_params(self, data):
    #     """
    #     Initialize distribution params
    #     """

    #     # init weights
    #     self.weights = np.ones(self.k) / self.k

    #     indexes = np.random.choice(data.shape[0], self.k, replace=False)
    #     # init mus
    #     self.mus = data[indexes].reshape(self.k)

    #     # init sigmas
    #     self.sigmas = np.random.random_integers(self.k)
    def init_params(self, data):
        """
        Initialize distribution params
        """
        # init weights
        self.weights = np.ones(self.k) / self.k

        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        # init mus
        self.mus = data[indexes].reshape(self.k)

        # init sigmas with small positive values
        self.sigmas = np.random.uniform(low=0.1, high=2.0, size=self.k)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """

        # calculate the res
        res = self.weights * norm_pdf(data.reshape(-1, 1), self.mus, self.sigmas)  # nopep8

        # calculate the sum
        sum_res = np.sum(res, axis=1, keepdims=True)

        # calculate the responsibilities
        self.responsibilities = res / sum_res

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """

        # calculate the new weights
        self.weights = np.mean(np.asarray(self.responsibilities), axis=0)

        # calculate the new mus
        self.mus = np.sum(data.reshape(-1, 1) * self.responsibilities, axis=0) / np.sum(np.asarray(self.responsibilities), axis=0)  # nopep8

        # calculate the new sigmas
        self.sigmas = np.sqrt(np.sum(self.responsibilities * (data.reshape(-1, 1) - self.mus) ** 2, axis=0) / np.sum(np.asarray(self.responsibilities), axis=0))  # nopep8

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """

        self.init_params(data)

        i = 0
        J_prev = 0
        self.costs = []

        while i < self.n_iter:
            self.expectation(data)
            self.maximization(data)

            J = self._cost_function(data)
            self.costs.append(J)

            if np.abs(J - J_prev) < self.eps:
                break

            J_prev = J
            i += 1

    def _cost_function(self, data):
        """
        Calculate the cost function for the EM algorithm
        """
        J = -np.sum(np.log(np.sum(self.weights * norm_pdf(data, self.mus, self.sigmas), axis=1)))  # nopep8

        return J

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    pdf = np.sum(weights * norm_pdf(data, mus, sigmas), axis=1)
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = []
        self.models = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """

        # get the unique classes
        classes = np.unique(y)
        self.prior = np.zeros(len(classes))
        n_features = X.shape[1]

        # loop over the classes, and then loop over the features, and train a model for each feature
        for i, c in enumerate(classes):
            # calculate the prior
            self.prior[i] = np.sum(y == c) / len(y)
            for j in range(n_features):
                # get the data for the current class and feature
                data = X[y == c, j]
                # train the model
                model = EM(k=self.k, random_state=self.random_state)
                model.fit(data)
                self.models[(c, j)] = model

    def _calc_likelihood(self, data, c):
        """
        Calculate the likelihood of the data for a given class
        """
        likelihood = 1
        n_features = data.shape[1]

        for j in range(n_features):
            model = self.models[(c, j)]
            weights, mus, sigmas = model.get_dist_params()
            likelihood *= gmm_pdf(data[:, j], weights, mus, sigmas)

        return likelihood

    def _calc_class_posterior(self, data, c):
        """
        Calculate the posterior of the data for a given class
        """
        posterior = self.prior[c] * self._calc_likelihood(data, c)
        return posterior

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None

        n_classes = len(self.prior)
        n_examples = X.shape[0]
        posteriors = np.zeros((n_classes, n_examples))

        for i in range(n_classes):
            posteriors[i] = self._calc_class_posterior(X, i)

        preds = np.argmax(posteriors, axis=0)

        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # 1. Logistic Regression
    lr = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lr.fit(x_train, y_train)
    preds = lr.predict(x_train)
    lor_train_acc = np.mean(preds == y_train)
    preds = lr.predict(x_test)
    lor_test_acc = np.mean(preds == y_test)

    # 2. Naive Bayes
    nb = NaiveBayesGaussian(k=k)
    nb.fit(x_train, y_train)
    preds = nb.predict(x_train)
    bayes_train_acc = np.mean(preds == y_train)
    preds = nb.predict(x_test)
    bayes_test_acc = np.mean(preds == y_test)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    """
    1. Generate a dataset (`dataset_a`), in 3 dimensions (3 features), with 2 classes, using **only** Multivariate-Gaussians (as many as you want) such that **Naive Bayes will work better on it when compared to Logisitc Regression**.
    2. Generate another dataset (`dataset_b`), in 3 dimensions (3 features), with 2 classes, using **only** Multivariate-Gaussians (as many as you want) such that **Logistic Regression will work better on it when compared to Naive Bayes**.
    3. Visualize the datasets: Plot one 3d graph.

    Make sure to use the "multivariate_normal" function from scipy.stats.
    """

    # dataset a
    np.random.seed(1991)
    mu1 = np.array([0, 0, 0])
    cov1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mu2 = np.array([1, 1, 1])
    cov2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    dataset_a_features = np.concatenate([multivariate_normal.rvs(mean=mu1, cov=cov1, size=100), multivariate_normal.rvs(mean=mu2, cov=cov2, size=100)])  # nopep8
    dataset_a_labels = np.concatenate([np.zeros(100), np.ones(100)])

    # dataset b
    mu1 = np.array([0, 0, 0])
    cov1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mu2 = np.array([1, 1, 1])
    cov2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    dataset_b_features = np.concatenate([multivariate_normal.rvs(mean=mu1, cov=cov1, size=100), multivariate_normal.rvs(mean=mu2, cov=cov2, size=100)])  # nopep8
    dataset_b_labels = np.concatenate([np.zeros(100), np.ones(100)])

    import matplotlib.pyplot as plt

    # plot the data, colors: 0 - blue, 1 - red
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset_a_features[dataset_a_labels == 0, 0], dataset_a_features[dataset_a_labels == 0, 1], dataset_a_features[dataset_a_labels == 0, 2], c='b', label='0')  # nopep8
    ax.scatter(dataset_a_features[dataset_a_labels == 1, 0], dataset_a_features[dataset_a_labels == 1, 1], dataset_a_features[dataset_a_labels == 1, 2], c='r', label='1')  # nopep8
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()

    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }
