import numpy as np

from mcmc import slice_sample

class SliceSampler(object):
    """generate samples from a model using slice sampling
    """

    def __init__(self, prior_mean, prior_var, thinning, init_params):
        self.prior_mean = prior_mean
        self.prior_sig = np.diag(prior_var)
        self.prior_const = 0.5 * (len(prior_mean)*np.log(2.*np.pi) + np.log(prior_var).sum())
        self.thinning = thinning
        self.params = np.copy(init_params)
        self.bound = (prior_mean- 3.* np.sqrt(prior_var), prior_mean + 3. * np.sqrt(prior_var))
        self.prev_x = None

    def logprob(self, x, gp_likelihood):
        """compute the log probability of observations x

        This includes the model likelihood as well as any prior
        probability of the parameters

        Returns
        -------
        lp : float
            the log probability
        """
        if np.any(np.greater(x, self.bound[1])) or np.any(np.less(x, self.bound[0])):
            print "Error: param out of bound, use prev x"
            x = self.prev_x
        self.prev_x = np.copy(x)
        x_mu = (x - self.prior_mean).reshape((-1, 1))

        gp_likelihood.set_hyperparameters(x)
        try:
            result = gp_likelihood.compute_log_likelihood() - 0.5 * np.dot(x_mu.T, np.dot(self.prior_sig, x_mu)) + self.prior_const
            return result
        except ValueError as e:
            print "ValueError: " + str(e)
            print "x {0}".format(x)
            return None

    def sample(self, gp_likelihood):
        """generate a new sample of parameters for the model

        Notes
        -----
        The parameters are stored as self.params which is a list of Params objects.
        The values of the parameters are updated on each call.  Pesumably the value of
        the parameter affects the model (this is not required, but it would be a bit
        pointless othewise)

        """
        # turn self.params into a 1d numpy array
        for i in xrange(self.thinning + 1):
            # get a new value for the parameter array via slice sampling
            self.params, current_ll = slice_sample(self.params, self.logprob, gp_likelihood, step_out=False)
        self.current_ll = current_ll # for diagnostics
        return self.params.copy()
