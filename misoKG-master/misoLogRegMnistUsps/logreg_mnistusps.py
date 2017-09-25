import random
import sys
import numpy

sys.path.append("../")

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla

import six.moves.cPickle as pickle
import gzip
import timeit

import theano
import theano.tensor as T

__author__ = 'matthiaspoloczek'

'''
The logistic regression benchmark on MNIST data (truth IS) that uses USPS data as additional IS
'''


class LogRegMNISTUSPS(RosenbrockVanilla):
    def __init__(self, mult=1.0, pathToData='/home/ubuntu/data'):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing,
        and default is minimizing
        :param pathToData: absolute path to folder that contains MNIST and USPS data
        """

        """
        Parameters:
        Learning rate, L2 penalty, batch size, #epoches
        """
        self._search_domain = numpy.array([
            [numpy.log(1e-10), numpy.log(1.0)],
            [0.0, 1.1],
            [20., 2000.],
            [5.,2000.]
        ])
        self._dim = 4               # dim of the domain
        self._outputStatus = False  # print frequent updates about status?

        self._num_IS = 2
        self._mult = mult
        self._func_name = 'lrMU'
        self._list_IS_to_query = [0,1]
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG

        self._meanValuesIS = [-0.97599, -0.97599]
        # IS 0 is centered around 0.0, IS 1 may have bias
        # IS 0 testscore mean = 0.279772673631
        # IS 0 logit mean = -0.975986708719
        # IS 1 testscore mean = 0.515932032472 ### repeated once more only for 100 samples: Computed mean 0.52176184459 for IS 1
        #
        #
        ### pure testscore:
        ### USPS set without test set and without log( ), thus the value for MNIST is the same as before:
        #[0.26749, 0.51683] # subtracted from evaluation to center the objective function at 0
        # Computed means for IS0, IS1: [0.26749110239434304, 0.51683456508187209], averaged over 1000 samples each
        # second run computed mean 0.517958533834 for IS 1

        # load datasets once here
        self._datasets_mnist = self.load_data(pathToData+'/mnist.pkl.gz')
        self._datasets_usps = self.load_data(pathToData+'/usps.pickle')

        # since it is common to report the validation error, should we also compute the test error?
        self._testModel = True

    def load_data(self, dataset):
        ''' Loads the dataset

        :type dataset: string
        :param dataset: the path to the dataset (here MNIST)
        '''

        #############
        # LOAD DATA #
        #############

        # # Download the MNIST dataset if it is not present
        # data_dir, data_file = os.path.split(dataset)
        # if data_dir == "" and not os.path.isfile(dataset):
        #     # Check if dataset is in the data directory.
        #     new_path = os.path.join(
        #         os.path.split(__file__)[0],
        #         "..",
        #         "data",
        #         dataset
        #     )
        #     if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
        #         dataset = new_path
        #
        # if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        #     from six.moves import urllib
        #     origin = (
        #         'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        #     )
        #     print('Downloading data from %s' % origin)
        #     urllib.request.urlretrieve(origin, dataset)

        if 'mnist' in dataset:
            # Load the MNIST dataset
            if self._outputStatus:
                print('... loading MNIST data')
            with gzip.open(dataset, 'rb') as f:
                try:
                    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
                except:
                    train_set, valid_set, test_set = pickle.load(f)
        elif 'usps' in dataset:
            if self._outputStatus:
                print('... loading USPS data')
            with open(dataset, "rb") as input_file:
                train_set, valid_set, test_set = pickle.load(input_file)

        # for int in train_set[1]:
        #     if (int < 0) or (int > 9):
        #         print int
        #         exit(0)

        # # Load the MNIST dataset
        # with gzip.open(dataset, 'rb') as f:
        #     try:
        #         train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        #     except:
        #         train_set, valid_set, test_set = pickle.load(f)

        # train_set, valid_set, test_set format: tuple(input, target)
        # input is a numpy.ndarray of 2 dimensions (a matrix)
        # where each row corresponds to an example. target is a
        # numpy.ndarray of 1 dimension (vector) that has the same length as
        # the number of rows in the input. It should give the target
        # to the example with the same index in the input.

        test_set_x, test_set_y = self.shared_dataset(test_set)
        valid_set_x, valid_set_y = self.shared_dataset(valid_set)
        train_set_x, train_set_y = self.shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval

    def shared_dataset(self, data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    def noise_and_cost_func(self, IS, x):
        '''
        Return the observational noise and the cost for each point at any IS

        Args:
            IS: the IS to be invoked
            x: the point

        Returns: a tuple (noise, cost)
        '''

        #IS 0 has mean variance 0.000000 and mean cost 36.416090 sec
        #IS 1 has mean variance 0.000000 and mean cost 11.713086 sec
        #
        #IS 0 has mean variance 0.000000. The cost have mean 32.226172 sec and variance 332.685530
        #IS 1 has mean variance 0.000000. The cost have mean 7.723285 sec and variance 27.113300

        # set noise to tiny value for numerical stability during matrix inversion
        if IS == 0:
            return 1e-6, 43.69 # taken from my estimation for the MISO paper
        elif IS == 1:
            return 1e-6, 10.42 # taken from my estimation for the MISO paper
        else:
            raise RuntimeError("illegal IS")

    def negative_log_likelihood(self, y, p_y_given_x):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        :type p_y_given_x:
        :param p_y_given_x:

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y, y_pred):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        :type y_pred:
        :param y_pred:
        """

        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(y_pred, y))
        else:
            raise NotImplementedError()

    def computeValue(self, IS, x):
        """
        # Run the logistic regression at info src IS with parameters give by x
        # return the mean test error
        :param IS: index of information source: 1, ..., M
        :param x: 4d numpy array
        :return: the obj value at x estimated by info source IS
        """

        # The implementation is an adaption of http://deeplearning.net/tutorial/logreg.html

        # :type n_in: int
        # :param n_in: number of input units, the dimension of the space in
        #              which the datapoints lie
        if IS == 0:
            train_set_x, train_set_y = self._datasets_mnist[0]
            valid_set_x, valid_set_y = self._datasets_mnist[1]
            test_set_x, test_set_y = self._datasets_mnist[2]
            n_in = 28 * 28 # number of pixels/features per example, i.e. train_set_x.shape[1]
        elif IS == 1:
            train_set_x, train_set_y = self._datasets_usps[0]
            valid_set_x, valid_set_y = self._datasets_usps[1]
            test_set_x, test_set_y = self._datasets_usps[2]
            n_in = 16 * 16
        else:
            raise RuntimeError("illegal IS")

        # :type n_classes: int
        # :param n_classes: number of output units, the dimension of the space in which the labels lie
        # since we are learning digits 0,...,9, the value is 10.
        n_classes = 10

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W = theano.shared(value=numpy.zeros((n_in, n_classes), dtype=theano.config.floatX),
                          name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        b = theano.shared(value=numpy.zeros((n_classes,), dtype=theano.config.floatX),
                          name='b', borrow=True)

        # keep track of model input and output
        matrix_x = T.matrix('x')
        vector_y = T.ivector('y')

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        p_y_given_x = T.nnet.softmax(T.dot(matrix_x, W) + b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        y_pred = T.argmax(p_y_given_x, axis=1)

        # Define L2 norm
        l2_sqr = (W ** 2).sum()

        # extract current point under evaluation
        learning_rate = numpy.float32(numpy.exp(x[0])) # the learning_rate was stored as log
        l2_reg = numpy.float32(x[1])
        batch_size = numpy.int32(x[2])
        n_epochs = numpy.int32(x[3])

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # the cost we minimize during training is the negative log likelihood + L2 regularization of
        # the model in symbolic format
        cost = self.negative_log_likelihood(vector_y, p_y_given_x) + l2_reg * l2_sqr

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
        if self._testModel:
            test_model = theano.function(inputs=[index],
                                         outputs=self.errors(vector_y, y_pred),
                                         givens={matrix_x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                                 vector_y: test_set_y[index * batch_size: (index + 1) * batch_size]
                                                 })
        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors(vector_y, y_pred),
            givens={matrix_x: valid_set_x[index * batch_size: \
                (index + 1) * batch_size],
                    vector_y: valid_set_y[index * batch_size: \
                        (index + 1) * batch_size]
                    })

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=W)
        g_b = T.grad(cost=cost, wrt=b)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(W, W - learning_rate * g_W), (b, b - learning_rate * g_b)]

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                matrix_x: train_set_x[index * batch_size: (index + 1) * batch_size],
                vector_y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )



        ###############
        # TRAIN MODEL #
        ###############
        if self._outputStatus:
            print('... training the model')
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        # go through this many minibatches before checking the network on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1

            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    if self._outputStatus:
                        print(
                            'epoch %i, minibatch %i/%i, validation error %f %%' %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                this_validation_loss * 100.
                            )
                        )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set
                        if self._testModel:
                            test_losses = [test_model(i) for i in range(n_test_batches)]
                            test_score = numpy.mean(test_losses)

                            if self._outputStatus:
                                print(
                                    (
                                        '     epoch %i, minibatch %i/%i, test error of'
                                        ' best model %f %%'
                                    ) %
                                    (
                                        epoch,
                                        minibatch_index + 1,
                                        n_train_batches,
                                        test_score * 100.
                                    )
                                )

                        # # save the best model
                        # with open('best_model.pkl', 'wb') as f:
                        #     pickle.dump(classifier, f)

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        if self._outputStatus:
            if self._testModel:
                print(
                    (
                        'Optimization complete with best validation score of %f %%, '
                        'with test performance %f %%'
                    )
                    % (best_validation_loss * 100., test_score * 100.)
                )
            else:
                print(
                    (
                        'Optimization complete with best validation score of %f %%.'
                    )
                    % (best_validation_loss * 100.)
                )
            print('The code run for %d epochs, with %f epochs/sec and %f sec in total' % (
                epoch, 1. * epoch / (end_time - start_time), (end_time - start_time)))
            # print(('The code for file ' +
            #        os.path.split(__file__)[1] +
            #        ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

        ### before transforming to logit, catch the extreme cases that the testscore is 0 or 1
        if test_score >= 1.0:
            logit_testscore = numpy.finfo(numpy.float64).max
        elif test_score <= 0.0:
            logit_testscore = numpy.finfo(numpy.float64).min
        else:
        ### Snoeck et al report the validation error, Hutter et al report the test error
            logit_testscore = numpy.log(test_score) - numpy.log(1.0 - test_score) # return the logit of the mean testscore
        #return numpy.log(float(test_score))
        return logit_testscore

    def evaluate(self, IS, x):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 8D numpy array
        :return the logit of the test score minus the mean logit
        """
        if IS in self._list_IS_to_query:
            x = self.ensureBoundaries(x) # address numerical issues arising in EGO
            return self._mult * (self.computeValue(IS, x) - self.getMeanValue(IS))
        else:
            raise RuntimeError("illegal IS")

    def estimateMeanFromPickles(self):
        '''
        Take the samples used to determine the hypers (or the training data) to estimate the mean value

        :return: the mean objective value across the samples (the sign assumes it is a minimization problem)
        '''

        raise RuntimeError("Not Implemented for new Interface")

    def estimateVariance(self, num_random_points = 5, num_samples_per_point = 10):
        '''
        estimate variance of each information source by querying each at the
        same 5 random positions via 10 samples
        :param num_random_points: the number of random points queried for each IS
        :param num_samples_per_point: the number of samples for each point
        :return: list of vars
        '''

        measured_objvalues =  numpy.zeros(num_samples_per_point)
        variance_objvalues =  numpy.zeros( num_random_points )

        for i in xrange(num_random_points):
            x = self.drawRandomPoint()

            for it in xrange(num_samples_per_point):
                measured_objvalues[it] = self.evaluate(0,x)

            # estimate variance and cost for this point
            variance_objvalues[i] = numpy.var(measured_objvalues)

        print variance_objvalues

    def estimateVarianceAndCost(self, num_random_points = 10, num_samples_per_point = 2):
        '''
        estimate variance and cost of each information source by querying each at the
        same random positions
        :param num_random_points: the number of random points queried for each IS
        :param num_samples_per_point: the number of samples for each point
        :return: None
        '''

        highestIndex = max(self.getList_IS_to_query())
        # print 'highestISIndex = ' + str(highestIndex)
        measured_objvalues =  numpy.zeros(num_samples_per_point)
        measured_evalcosts =  numpy.zeros(num_samples_per_point)
        variance_objvalues =  numpy.zeros( (num_random_points, highestIndex+1) )
        mean_evalcosts =  numpy.zeros( (num_random_points, highestIndex+1) )

        for i in xrange(num_random_points):
            x = self.drawRandomPoint()
            for IS in self.getList_IS_to_query(): # IS+1 is the info source
                # query each specified IS at x

                for it in xrange(num_samples_per_point):
                    start_time = timeit.default_timer()
                    measured_objvalues[it] = self.evaluate(IS,x)
                    elapsed_time = timeit.default_timer() - start_time
                    measured_evalcosts[it] = elapsed_time

                # estimate variance and cost for this point
                variance_objvalues[i][IS] = numpy.var(measured_objvalues)
                mean_evalcosts[i][IS] = numpy.mean(measured_evalcosts)

        for IS in self.getList_IS_to_query(): # IS+1 is the info source
            print (('IS %d has mean variance %f. The cost have mean %f sec and std dev %f') %
            (IS, numpy.mean(variance_objvalues,axis=0)[IS], numpy.mean(mean_evalcosts,axis=0)[IS],
             numpy.std(mean_evalcosts,axis=0)[IS])
                   )

    def getList_IS_to_query(self):
        return self._list_IS_to_query

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])


'''
LogRegMNISTUSPS2 differs from LogRegMNISTUSPS in that IS1 is also centered at 0.0
'''

class LogRegMNISTUSPS2(LogRegMNISTUSPS):
    def __init__(self, mult=1.0, pathToData='/home/ubuntu/data'):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing,
        and default is minimizing
        :param pathToData: absolute path to folder that contains MNIST and USPS data
        """

        """
        Parameters:
        Learning rate, L2 penalty, batch size, #epoches
        """
        self._search_domain = numpy.array([
            [numpy.log(1e-10), numpy.log(1.0)],
            [0.0, 1.1],
            [20., 2000.],
            [5.,2000.]
        ])
        self._dim = 4               # dim of the domain
        self._outputStatus = False  # print frequent updates about status?

        self._num_IS = 2
        self._mult = mult
        self._func_name = 'lrMU2'
        self._list_IS_to_query = [0,1]
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG

        self._meanValuesIS = [-0.97599, 0.01502]
        # IS 0 is centered around 0.0, IS 1 is now also for this variant of lrMU

        # load datasets once here
        self._datasets_mnist = self.load_data(pathToData+'/mnist.pkl.gz')
        self._datasets_usps = self.load_data(pathToData+'/usps.pickle')

        # since it is common to report the validation error, should we also compute the test error?
        self._testModel = True



'''
LogRegMNISTUSPS3 differs from LogRegMNISTUSPS2 in the cost for IS1

Note that it uses the data for LogRegMNISTUSPS2 (=lrMU2) to estimate hypers, and also loads the estimated hypers and
initial data of lrMU2.

See miso_lrMU3.py
'''

class LogRegMNISTUSPS3(LogRegMNISTUSPS2):
    def __init__(self, mult=1.0, pathToData='/home/ubuntu/data'):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing,
        and default is minimizing
        :param pathToData: absolute path to folder that contains MNIST and USPS data
        """

        """
        Parameters:
        Learning rate, L2 penalty, batch size, #epoches
        """
        self._search_domain = numpy.array([
            [numpy.log(1e-10), numpy.log(1.0)],
            [0.0, 1.1],
            [20., 2000.],
            [5.,2000.]
        ])
        self._dim = 4               # dim of the domain
        self._outputStatus = False  # print frequent updates about status?

        self._num_IS = 2
        self._mult = mult
        self._func_name = 'lrMU3'
        self._list_IS_to_query = [0,1]
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG

        self._meanValuesIS = [-0.97599, 0.01502]
        # IS 0 is centered around 0.0, IS 1 is now also for this variant of lrMU

        # load datasets once here
        self._datasets_mnist = self.load_data(pathToData+'/mnist.pkl.gz')
        self._datasets_usps = self.load_data(pathToData+'/usps.pickle')

        # since it is common to report the validation error, should we also compute the test error?
        self._testModel = True

    def noise_and_cost_func(self, IS, x):
        '''
        Return the observational noise and the cost for each point at any IS

        Args:
            IS: the IS to be invoked
            x: the point

        Returns: a tuple (noise, cost)
        '''

        # IS0 uses 28*28 pixels per image, IS1 only 16*16. Moreover, IS1 has about 5000 images for training, IS0 uses 60,000.
        # However, the actual cost depends on the hardware, the version of Theano (up to 30%), and the system load.
        # Thus, I am assuming a cost ratio IS0/IS1 of 20 instead of 3*12 = 36.

        # set noise to tiny value for numerical stability during matrix inversion
        if IS == 0:
            return 1e-6, 43.69 # taken from my estimation for the MISO paper
        elif IS == 1:
            return 1e-6, 2.18 # taken from my estimation for the MISO paper
        else:
            raise RuntimeError("illegal IS")


'''
LogRegMNISTUSPS4 differs from LogRegMNISTUSPS3 in the cost for IS1

Note that it uses the data for LogRegMNISTUSPS2 (=lrMU2) to estimate hypers, and also loads the estimated hypers and
initial data of lrMU2.

See miso_lrMU4.py
'''

class LogRegMNISTUSPS4(LogRegMNISTUSPS2):
    def __init__(self, mult=1.0, pathToData='/home/ubuntu/data'):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing,
        and default is minimizing
        :param pathToData: absolute path to folder that contains MNIST and USPS data
        """

        """
        Parameters:
        Learning rate, L2 penalty, batch size, #epoches
        """
        self._search_domain = numpy.array([
            [numpy.log(1e-10), numpy.log(1.0)],
            [0.0, 1.1],
            [20., 2000.],
            [5.,2000.]
        ])
        self._dim = 4               # dim of the domain
        self._outputStatus = False  # print frequent updates about status?

        self._num_IS = 2
        self._mult = mult
        self._func_name = 'lrMU4'
        self._list_IS_to_query = [0,1]
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG

        self._meanValuesIS = [-0.97599, 0.01502]
        # IS 0 is centered around 0.0, IS 1 is now also for this variant of lrMU

        # load datasets once here
        self._datasets_mnist = self.load_data(pathToData+'/mnist.pkl.gz')
        self._datasets_usps = self.load_data(pathToData+'/usps.pickle')

        # since it is common to report the validation error, should we also compute the test error?
        self._testModel = True

    def noise_and_cost_func(self, IS, x):
        '''
        Return the observational noise and the cost for each point at any IS

        Args:
            IS: the IS to be invoked
            x: the point

        Returns: a tuple (noise, cost)
        '''

        # IS0 uses 28*28 pixels per image, IS1 only 16*16. Moreover, IS1 has about 5000 images for training, IS0 uses 60,000.
        # However, the actual cost depends on the hardware, the version of Theano (up to 30%), and the system load.
        # Thus, I am assuming a cost ratio IS0/IS1 of 10 instead of 3*12 = 36.

        # set noise to tiny value for numerical stability during matrix inversion
        if IS == 0:
            return 1e-6, 43.69 # taken from my estimation for the MISO paper
        elif IS == 1:
            return 1e-6, 4.5 # taken from my estimation for the MISO paper
        else:
            raise RuntimeError("illegal IS")