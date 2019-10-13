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

def initialize_weights(nweights):
    import numpy as np
    return(np.random.uniform(size=(nweights, 1)))

def lin_reg_loss(X, y, weights): # MSE
    import numpy as np
    p = X @ weights
    #p = np.round(X @ weights)
    return(1/len(y) * np.sum(p**2 - 2*y*p + y**2))

def lin_reg_grad(X, y, weights):
    import numpy as np
    p = X @ weights
    #p = np.round(X @ weights)
    return(2/len(y) * (X.T @ (p-y)))

def normal_eqn(X, y):
    import numpy as np
    #print(np.linalg.inv(X.T@X)@X.T@y)
    return(np.linalg.lstsq(X, y, rcond=None)[0])

def output_generalization_error(X_test, y_test, weights, loss_func):
    import numpy as np
    sorted_ind = np.argsort(y_test.flatten())
    X_test = X_test[sorted_ind,:]
    y_test = y_test[sorted_ind]
    print(np.array(np.round(np.concatenate((y_test, X_test@weights), axis=1)), dtype='uint8'))
    print('Generalization error:', loss_func(X_test, y_test, weights))

def plot_loss(losses_train, losses_valid):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.plot(losses_train, label='Training')
    ax.plot(losses_valid, label='Validation')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.show()

def run_epoch(X_train, y_train, X_valid, y_valid, weights, loss_func, grad_func, lr, batch_size):
    import numpy as np

    # Get the number of datapoints and batch size
    m = len(y_train)
    if batch_size == None:
        batch_size = m

    # Shuffle the datapoints
    order = np.array(range(m))
    np.random.shuffle(order)
    X_train = X_train[order,:]
    y_train = y_train[order]

    # Calculate the batch sizes
    nbatches = int(np.ceil(m/batch_size))
    batch_sizes = [batch_size]*(nbatches-1)
    last_batch = m % batch_size
    if last_batch == 0:
        last_batch = batch_size
    batch_sizes = np.array(np.concatenate((batch_sizes, [last_batch])), dtype='uint32')

    # Update the weights once per batch
    data_start = 0
    for _, batch_size0 in enumerate(batch_sizes):
        X_train0 = X_train[data_start:data_start+batch_size0,:]
        y_train0 = y_train[data_start:data_start+batch_size0]
        grad = grad_func(X_train0, y_train0, weights)
        weights = weights - lr*grad
        loss_train = loss_func(X_train, y_train, weights)
        loss_valid = loss_func(X_valid, y_valid, weights)
        data_start = data_start + batch_size0
        #print('  After batch {}/{}: loss_train={}, loss_valid={}'.format(ibatch+1, nbatches, loss_train, loss_valid))

    return(weights, loss_train, loss_valid)

def train(X, y, weights, loss_func, grad_func, nepochs, lr, batch_size, validation_perc):

    import numpy as np

    # Hold out some data as a validation set
    m = len(y)
    m_train = m - int(np.round(validation_perc/100*m))
    X_train = X[:m_train,:]
    y_train = y[:m_train]
    X_valid = X[m_train:,:]
    y_valid = y[m_train:]

    # Training on the training set while computing the loss on the validation set as well
    losses_train = []
    losses_valid = []
    for iepoch in range(nepochs):
        weights, loss_train, loss_valid = run_epoch(X_train, y_train, X_valid, y_valid, weights, loss_func, grad_func, lr, batch_size)
        losses_train.append(loss_train)
        losses_valid.append(loss_valid)
        print('After epoch {}/{}: loss_train={}, loss_valid={}'.format(iepoch+1, nepochs, loss_train, loss_valid))
    return(weights, losses_train, losses_valid)


# Inputs
data_func = get_iris_data
model = (lin_reg_loss, lin_reg_grad)
nepochs = 500
lr = 0.005
batch_size = 10
rand_seed = 44
test_perc = 20
validation_perc = 20

# Load the data and get it into appropriate form
features, labels = data_func()
X_train, y_train, X_test, y_test = get_arrays(features, labels, rand_seed, test_perc)

# Initialize the weights
weights = initialize_weights(X_train.shape[1])

# Train the model
weights, losses_train, losses_valid = train(X_train, y_train, weights, model[0], model[1], nepochs, lr, batch_size, validation_perc)

# Output the results
print('Weights:', weights)
plot_loss(losses_train, losses_valid)
print('Analytic weights:', normal_eqn(X_train, y_train))
output_generalization_error(X_test, y_test, weights, model[0])



def softmax_reg_loss(X, y, weights): # MSE
    import numpy as np


    # Pick up here!!!!


    p = X @ weights
    #p = np.round(X @ weights)
    return(1/len(y) * np.sum(p**2 - 2*y*p + y**2))

