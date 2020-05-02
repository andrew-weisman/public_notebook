class Common:

    def __init__(self, nepochs, lr, batch_size, validation_perc, weights=None):
        self.weights = weights
        self.nepochs = nepochs
        self.lr = lr
        self.batch_size = batch_size
        self.validation_perc = validation_perc

    # Dummy methods to make linter happy
    def init_weights(self, X, y):
        return(0)
    def calc_grad(self, X, y):
        return(0)
    def calc_loss(self, X, y):
        return(0)

    def fit(self, X, y):
        import numpy as np

        if self.weights == None:
            #self.weights = self.init_weights(X.shape)
            self.weights = self.init_weights(X, y)

        # Hold out some data as a validation set
        m = len(y)
        m_train = m - int(np.round(self.validation_perc/100*m))
        X_train = X[:m_train,:]
        y_train = y[:m_train]
        X_valid = X[m_train:,:]
        y_valid = y[m_train:]

        # Training on the training set while computing the loss on the validation set as well
        losses_train = []
        losses_valid = []
        for iepoch in range(self.nepochs):
            loss_train, loss_valid = self.run_epoch(X_train, y_train, X_valid, y_valid)
            losses_train.append(loss_train)
            losses_valid.append(loss_valid)
            print('After epoch {}/{}: loss_train={}, loss_valid={}'.format(iepoch+1, self.nepochs, loss_train, loss_valid))

        return(losses_train, losses_valid)

    def run_epoch(self, X_train, y_train, X_valid, y_valid):
        import numpy as np

        # Get the number of datapoints and batch size
        m = len(y_train)
        if self.batch_size == None:
            self.batch_size = m

        # Shuffle the datapoints
        order = np.array(range(m))
        np.random.shuffle(order)
        X_train = X_train[order,:]
        y_train = y_train[order]

        # Calculate the batch sizes
        nbatches = int(np.ceil(m/self.batch_size))
        batch_sizes = [self.batch_size]*(nbatches-1)
        last_batch = m % self.batch_size
        if last_batch == 0:
            last_batch = self.batch_size
        batch_sizes = np.array(np.concatenate((batch_sizes, [last_batch])), dtype='uint32')

        # Update the weights once per batch
        data_start = 0
        for _, batch_size0 in enumerate(batch_sizes):
            X_train0 = X_train[data_start:data_start+batch_size0,:]
            y_train0 = y_train[data_start:data_start+batch_size0]
            grad = self.calc_grad(X_train0, y_train0)
            self.weights = self.weights - self.lr*grad
            loss_train = self.calc_loss(X_train, y_train)
            loss_valid = self.calc_loss(X_valid, y_valid)
            data_start = data_start + batch_size0
            #print('  After batch {}/{}: loss_train={}, loss_valid={}'.format(ibatch+1, nbatches, loss_train, loss_valid))

        return(loss_train, loss_valid)

    def sigmoid(self, t):
        import numpy as np
        return(1/(1+np.exp(-t)))

class LinReg(Common):

    def init_weights(self, X, y):
        import numpy as np
        sz = (X.shape[1],1)
        return(np.random.uniform(size=sz)*2-1)

    def calc_hyp(self, X):
        return(X @ self.weights)

    def calc_loss(self, X, y): # MSE
        import numpy as np
        p = self.calc_hyp(X)
        #return(1/len(y) * np.sum(p**2 - 2*y*p + y**2))
        return(1/len(y) * np.sum((p-y)**2, axis=0))

    def calc_grad(self, X, y):
        p = self.calc_hyp(X)
        return(2/len(y) * (X.T @ (p-y)))

    def predict(self, X):
        p = self.calc_hyp(X)
        return(p)

    def predict_class(self, X):
        import numpy as np
        p = self.calc_hyp(X)
        return(np.round(p))

class LogReg(Common):

    def init_weights(self, X, y):
        import numpy as np
        sz = (X.shape[1],1)
        return(np.random.uniform(size=sz)*2-1)

    def calc_hyp(self, X):
        return(self.sigmoid(X@self.weights))

    def calc_loss(self, X, y): # log loss (=binary cross entropy?)
        import numpy as np
        p = self.calc_hyp(X)
        return(-1/len(y) * np.sum( y*np.log(p) + (1-y)*np.log(1-p), axis=0) )

    def calc_grad(self, X, y):
        p = self.calc_hyp(X)
        return(1/len(y) * (X.T @ (p-y)))

    def predict(self, X):
        import numpy as np
        p = self.calc_hyp(X)
        return(np.round(p))

class SoftReg(Common):

    def init_weights(self, X, y):
        import numpy as np
        sz = (X.shape[1],len(np.unique(y)))
        return(np.random.uniform(size=sz)*2-1) # n x K (different)

    def score(self, X):
        return(X @ self.weights) # m x K

    def calc_hyp(self, X):
        import numpy as np
        numerator = np.exp(self.score(X)) # m x K
        K = numerator.shape[1]
        denominator = np.tile(np.sum(numerator, axis=1).reshape(-1,1), (1,K)) # m x K
        return(numerator/denominator) # m x K (different)

    def calc_loss(self, X, y): # cross entropy
        import numpy as np
        K = self.weights.shape[1]
        m = y.shape[0]
        y_mat = np.tile(y, (1,K)) # m x K
        k_mat = np.tile(np.arange(K).reshape((1,K)), (m,1)) # m x K
        y_mat = np.array(y_mat==k_mat, dtype='uint8') # m x K
        p_mat = self.calc_hyp(X) # m x K
        return(-1/m*np.sum(np.sum(y_mat * np.log(p_mat), axis=0), axis=0)) # 1 x 1 (same as usual)

    def calc_grad(self, X, y):
        import numpy as np
        K = self.weights.shape[1]
        m = y.shape[0]
        y_mat = np.tile(y, (1,K)) # m x K
        k_mat = np.tile(np.arange(K).reshape((1,K)), (m,1)) # m x K
        y_mat = np.array(y_mat==k_mat, dtype='uint8') # m x K
        p_mat = self.calc_hyp(X) # m x K
        return(1/m * (X.T @ (p_mat-y_mat))) # n x K (different)

    def predict(self, X):
        import numpy as np
        return(np.argmax(self.score(X), axis=1)) # m x 1 (same as usual)

def get_arrays(features, labels, rand_seed, test_perc):
    import numpy as np

    # Get the full X and y arrays
    m = len(labels)
    X = np.ones((m, 1))
    for feature in features:
        X = np.concatenate((X, feature.reshape((-1,1))), axis=1)
    y = labels.reshape((-1,1))

    # Set the random seed
    np.random.seed(rand_seed)

    # Shuffle the full arrays
    order = np.array(range(m))
    np.random.shuffle(order)
    X = X[order,:]
    y = y[order]

    # Determine the training and test sets
    m_train = m - int(np.round(test_perc/100*m))
    X_train = X[:m_train,:]
    y_train = y[:m_train]
    X_test = X[m_train:,:]
    y_test = y[m_train:]

    return(X_train, y_train, X_test, y_test)

def get_iris_data():
    from sklearn import datasets
    iris = datasets.load_iris()
    lengths = iris['data'][:,2] # of petal, in cm
    widths = iris['data'][:,3] # of petal, in cm
    labels = iris['target']
    return((lengths, widths), labels)

def make_binary(y, positive_class):
    import numpy as np
    return(np.array(y==positive_class, dtype='uint8'))

def normal_eqn(X, y):
    import numpy as np
    return(np.linalg.lstsq(X, y, rcond=None)[0])

def output_generalization_error(model, X_test, y_test, do_class=True):
    import numpy as np
    sorted_ind = np.argsort(y_test.flatten())
    X_test = X_test[sorted_ind,:]
    y_test = y_test[sorted_ind]
    if do_class:
        #print(np.array(np.concatenate((y_test, model.predict_class(X_test)), axis=1), dtype='uint8'))
        print(np.array(np.concatenate((y_test, model.predict_class(X_test).reshape((-1,1))), axis=1), dtype='uint8')) # if things fail in the future, maybe try using above line instead of this one
    else:
        print(np.array(np.concatenate((y_test, model.predict(X_test).reshape((-1,1))), axis=1), dtype='uint8'))
    print('Generalization error:', model.calc_loss(X_test, y_test))

def plot_loss(losses_train, losses_valid):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.plot(losses_train, label='Training')
    ax.plot(losses_valid, label='Validation')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.show()


def lin_reg_example():

    # Inputs
    data_func = get_iris_data
    Model = LinReg
    nepochs = 500
    lr = 0.005
    batch_size = 10
    rand_seed = 44
    test_perc = 20
    validation_perc = 20

    # Load the data and get it into appropriate form
    features, labels = data_func()
    X_train, y_train, X_test, y_test = get_arrays(features, labels, rand_seed, test_perc)

    # Initialize the regressor
    model = Model(nepochs, lr, batch_size, validation_perc)

    # Train the model
    _, _ = model.fit(X_train, y_train)

    # Output the results
    print('Weights:', model.weights)
    #plot_loss(losses_train, losses_valid)
    print('Analytic weights:', normal_eqn(X_train, y_train))
    output_generalization_error(model, X_test, y_test)

def log_reg_example(positive_class=0, rand_seed=44):

    # Inputs
    data_func = get_iris_data
    Model = LogReg
    nepochs = 500
    lr = 0.005
    #batch_size = 10
    batch_size = 10
    #rand_seed = 44
    test_perc = 20
    validation_perc = 20
    #positive_class = 0

    # Load the data and get it into appropriate form
    features, labels = data_func()
    X_train, y_train, X_test, y_test = get_arrays(features, labels, rand_seed, test_perc)

    # Make the categories binary
    y_train = make_binary(y_train, positive_class)
    y_test = make_binary(y_test, positive_class)

    # Initialize the regressor
    model = Model(nepochs, lr, batch_size, validation_perc)

    # Train the model
    _, _ = model.fit(X_train, y_train)

    # Output the results
    print('Weights:', model.weights)
    #plot_loss(losses_train, losses_valid)
    #print('Analytic weights:', normal_eqn(X_train, y_train))
    output_generalization_error(model, X_test, y_test, do_class=False)

    #return(y_train, y_test)

def soft_reg_example():

    from sklearn.linear_model import LogisticRegression

    # Inputs
    data_func = get_iris_data
    #Model = SoftReg
    Model = LogisticRegression
    nepochs = 2000
    lr = 0.010
    batch_size = 10
    rand_seed = 44
    test_perc = 20
    validation_perc = 20

    # Load the data and get it into appropriate form
    features, labels = data_func()
    X_train, y_train, X_test, y_test = get_arrays(features, labels, rand_seed, test_perc)

    # Initialize the regressor and train the model
    if Model == SoftReg:
        model = Model(nepochs, lr, batch_size, validation_perc)
        _, _ = model.fit(X_train, y_train)

        # Output the results
        print('Weights:', model.weights)
        #plot_loss(losses_train, losses_valid)
        #print('Analytic weights:', normal_eqn(X_train, y_train))
        output_generalization_error(model, X_test, y_test, do_class=False)
    else:
        model = Model(multi_class="multinomial", solver="lbfgs", C=10, verbose=2)
        model.fit(X_train, y_train)

        # Output the results (don't forget I'm not training on the validation data like they are)
        print(model.coef_)
        import numpy as np
        estimates = np.array(model.predict(X_test))
        y_test = y_test.flatten()
        sorted = np.argsort(y_test)
        print(y_test[sorted])
        print(estimates[sorted])
