import numpy as np
from .base import *
import logging
import tqdm
from scipy.optimize import minimize
import time
import warnings
from scipy.linalg import inv, det

logging.basicConfig(level=logging.INFO,
                    format='')
logger = logging.getLogger(__name__)

def initialize_parameters(
        m_prior = 1., 
        beta_prior = 1.,*,
        dtype, n_components, n_contexts):
    
    mu = np.zeros(n_components).astype(dtype)

    sigma = np.diag(np.ones(n_components)).astype(dtype)
    
    b = (np.ones((n_components,2,3)) * m_prior).astype(dtype, copy = False)
    
    nu = (np.ones((2, n_contexts//2)) * beta_prior).astype(dtype, copy = False)
    
    return b, nu, mu, sigma


def initialize_variational_parameters(*,random_state, dtype, n_components, n_contexts):
    
    
    _lambda = random_state.gamma(100, 1./100., 
                    (n_components, 2, n_contexts//2) # n_components, two mut categories, n_contexts
        ).astype(dtype, copy = False)
    
    
    epsilon = random_state.gamma(100., 1./100., 
                              (n_components, 2, n_contexts//2, 3) # n_components, two mut categories, n_contexts, n_mutations
                            ).astype(dtype, copy=False)
    
    return epsilon, _lambda


def initialize_document_variational_parameters(*,
        random_state, dtype, n_samples, n_components):

    gamma = random_state.normal(0.,0.1, 
                               (n_samples, n_components) # n_genomes by n_components
                              ).astype(dtype, copy=False)

    v = np.square(random_state.gamma(100., 1./100., 
                               (n_samples, n_components) # n_genomes by n_components
                              ).astype(dtype, copy=False))
    
    return gamma, v


def E_step_squigly(*,gamma, v):
    
    return np.sum(
        np.exp(gamma + v/2),
        axis = -1
    )


def E_step_gamma(*,weighted_phi, gamma, v, squigly,
                mu, inv_sigma, freq, tol = 1e-8):
    
    phisum = weighted_phi.sum(axis = (-3, -2, -1))
    
    def objective(gamma):

        E_nu = np.exp(gamma+v/2)
        mu_diff = (gamma - mu)

        ### Objective
        obj = -1/2 * mu_diff.dot(inv_sigma).dot(mu_diff[:,None])

        obj += np.sum(weighted_phi*gamma[:,None,None,None]) + \
            freq*\
            ( 1 - np.log(squigly) - 1/squigly*np.sum(E_nu, axis = -1))

        ### Jacobian
        jac = np.squeeze(-np.dot(inv_sigma, mu_diff[:,None]).T) + \
               phisum - (freq/squigly)*E_nu
        
        return -obj[0], -jac  # maximize!!!!


    def hess(gamma, p):
        hess = np.dot(inv_sigma, p) + np.dot( (freq/squigly)*np.exp(gamma + v/2) , p )
        return -hess

    #initial_loss = -objective(gamma)[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        new_gamma = minimize(
            objective, 
            gamma,
            jac = True,
            hessp = hess,
            method='newton-cg',
            tol=tol
        ).x
    
    #improvement = -objective(new_gamma)[0] - initial_loss
    improvement = None
    return new_gamma, improvement


def blei_E_step_v(*, gamma, v_sq, squigly, freq, tol = 1e-8,
                inv_sigma):

    #hessian_matrix = np.zeros((v_sq.shape, v_sq.shape))
    
    def objective(v_sq):
        
        E_nu = np.exp(gamma + v_sq/2)

        ### Objective
        obj = -1/2 * np.trace(np.diag(v_sq).dot(inv_sigma)) \
            + freq*( 1 - np.log(squigly) - 1/squigly * np.sum(E_nu, axis = -1)) \
            + 1/2 * np.sum(np.log(v_sq) + np.log(2*np.pi) + 1)

        ### Jacobian
        jac = -np.diag(inv_sigma)/2 \
            - freq/(2*squigly) * E_nu + 1/(2*v_sq)
        
        return -obj, -jac
    
    def hess(v_sq):

        hessian_matrix = np.fill_diagonal(
            hessian_matrix,
            -freq/(2*squigly)*np.exp(gamma + v_sq/2) - 1/(2*np.square(v_sq)),
        )
        return -hessian_matrix
    
    #initial_loss = -objective(v_sq)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        new_v = minimize(
            objective, 
            v_sq,
            jac = True,
            method='l-bfgs-b',
            tol = tol,
            bounds= [(1e-10, np.inf) for _ in range(v_sq.shape[-1])],
        ).x

    '''new_v = minimize(
        objective, 
        v_sq,
        jac = True,
        hess=hess,
        method='tnc',
        bounds= [(1e-10, np.inf) for _ in range(v_sq.shape[-1])],
    ).x'''
    
    #improvement = -objective(new_v)[0] - initial_loss
    improvement = None

    return new_v, improvement

def E_step_v(*, gamma, v_sq, squigly, freq, tol = 1e-8,
                inv_sigma):

    #hessian_matrix = np.zeros((v_sq.shape, v_sq.shape))
    
    def objective(r):

        v_sq = np.exp(2*r)
        E_nu = np.exp(gamma + v_sq/2)

        ### Objective
        obj = -1/2 * np.trace(np.diag(v_sq).dot(inv_sigma)) \
            + freq*( 1 - np.log(squigly) - 1/squigly * np.sum(E_nu, axis = -1)) \
            + 1/2 * np.sum(np.log(v_sq) + np.log(2*np.pi) + 1)

        ### Jacobian
        jac = -np.diag(inv_sigma)*v_sq \
            - freq/(2*squigly)*np.exp(gamma + 1/2*v_sq + 2*r) + 1
        
        return -obj, -jac
    
    def hess(r, p):
        
        v_sq = np.exp(2*r)

        hessp = (-2*np.diag(inv_sigma)*v_sq \
                - freq/(2*squigly)*np.exp(gamma + 1/2*v_sq + 2*r)*(v_sq + 2))*p

        return -hessp
    

    r0 = 1/2*np.log(v_sq)

    initial_loss = -objective(r0)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        new_r = minimize(
            objective, 
            r0,
            jac = True,
            hessp = hess,
            method='newton-cg',
            tol = tol,
        ).x

        '''new_r = minimize(
            objective, 
            r0,
            jac = True,
            #hessp = hess,
            method='l-bfgs-b',
            bounds = [(1e-10, np.inf) for _ in range(r0.shape[-1])],
            tol = tol,
        ).x'''
    
    improvement = -objective(new_r)[0] - initial_loss
    #improvement = None
    #print(improvement)

    new_v = np.exp(2*new_r)

    return new_v, improvement


def E_step_phi(gamma, phi_matrix_prebuild, freq_matrix):

        phi_matrix_unnormalized = phi_matrix_prebuild * np.exp(gamma)[:,:,None,None,None]
        phi_matrix = phi_matrix_unnormalized/phi_matrix_unnormalized.sum(1, keepdims = True)

        weighted_phi = phi_matrix*np.expand_dims(freq_matrix, 1)

        return phi_matrix, weighted_phi


class CorrelatedTopicModel(BaseModel):

    n_contexts = 32

    @property
    def sigma(self):
        return self._sigma

    @property
    def inv_sigma(self):
        return self._inv_sigma

    @sigma.setter
    def sigma(self, _sigma):
        self._sigma = _sigma
        self._inv_sigma = inv(_sigma)


    def M_step_mu_sigma(self, gamma, v, rho):
    
        mu_hat = gamma.mean(0)
        sigma_hat = 1/len(gamma) * ( (gamma - mu_hat).T.dot(gamma - mu_hat) \
            + np.array([np.diag(v[i]) for i in range(len(v))]).sum(0) )
        
        return self.svi_update(self.mu, mu_hat, rho), \
                self.svi_update(self.sigma, sigma_hat, rho)


    def _bound(self,*, gamma, v, phi_matrix, weighted_phi, freqs):

        logE_gamma = gamma # for consistency in bound equations

        evidence = 0
        evidence += np.sum(weighted_phi*(
                np.expand_dims(self.logE_epsilon, 0) + logE_gamma[:,:,None,None,None] \
                    + self.logE_lambda[None,:,:,:,None] - np.where(weighted_phi > 0, np.log(phi_matrix), 0.)
                )
            )

        for i in range(len(freqs)):

            evidence += 1/2 * np.log(det(self.inv_sigma)) - self.n_components/2*np.log(2*np.pi) \
                 -1/2 * (np.trace(np.diag(v[i]).dot(self.inv_sigma)) + (gamma[i] - self.mu).dot(self.inv_sigma).dot((gamma[i] - self.mu)))

            evidence += -freqs[i]*np.log(np.sum(np.exp(gamma[i]+v[i]/2), axis = -1))

            evidence += 1/2 * np.sum(np.log(v[i]) + np.log(2*np.pi) + 1)
        
        for i in range(self.n_components):
            for j in [0,1]:
                evidence+=sum(
                    dirichlet_bound(self.b[i,j], self.epsilon[i,j], self.logE_epsilon[i,j])
                )
        
        evidence+=sum(
            dirichlet_bound(
                self.nu.reshape(-1), 
                self._lambda.reshape(self.n_components,-1), 
                self.logE_lambda.reshape(self.n_components,-1)
            )
        )

        return evidence


    def _inference(self,*, gamma, v, freq_matrix, freqs, difference_tol = 1e-2, iterations = 100,
                quiet = True):

        phi_matrix_prebuild = np.exp(self.logE_lambda)[:,:,:,None] * np.exp(self.logE_epsilon)

        phis, weighted_phis = [],[]
        not_converged = 0

        for i in tqdm.tqdm(
            range(len(gamma)), 
            desc = 'Inferring latent variables' if not quiet else '\tE-step progress',
            ncols=50,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
            ): # outer data loop 
            
            for j in range(iterations): # inner convergence loop

                old_gamma = gamma[i].copy()

                squigly = E_step_squigly(gamma = gamma[i], v = v[i])
                
                phi_matrix_unnormalized = phi_matrix_prebuild * np.exp(gamma[i,:,None,None,None])
                phi_matrix = phi_matrix_unnormalized/phi_matrix_unnormalized.sum(0, keepdims = True)
                
                weighted_phi = phi_matrix*np.expand_dims(freq_matrix[i], 0)

                gamma[i], _ = E_step_gamma(weighted_phi = weighted_phi,
                                            gamma = gamma[i], 
                                            v = v[i], 
                                            squigly = squigly,
                                            freq=freqs[i],
                                            mu = self.mu, 
                                            inv_sigma = self.inv_sigma, 
                                            tol = difference_tol/100
                                        )

                v[i], _ = E_step_v(freq = freqs[i],
                                    gamma = gamma[i], 
                                    v_sq = v[i], 
                                    squigly = squigly,
                                    inv_sigma = self.inv_sigma,
                                    tol = difference_tol/100
                                 )


                mean_gamma_diff = np.abs(gamma[i] - old_gamma).mean()

                if mean_gamma_diff < difference_tol:
                    break

            else:
                not_converged+=1
            
            phis.append(phi_matrix)
            weighted_phis.append(weighted_phi)
        
        if not_converged > 0:
            logging.debug('\t{} samples reached maximum E-step iterations.'\
                    .format(str(not_converged)))

        return gamma, v, np.array(phis), np.array(weighted_phis)


    def get_freqs(self, freq_matrix):
        return freq_matrix.reshape((len(freq_matrix), -1)).sum(-1)
    

    @extract_freqmatrix
    def fit(self, freq_matrix):

        assert isinstance(self.n_components, int) and self.n_components > 1

        self.random_state = np.random.RandomState(self.seed)

        self.b, self.nu, self.mu, self.sigma = initialize_parameters(
            m_prior= self.m_prior, 
            beta_prior= self.beta_prior,
            dtype = self.dtype,
            n_components = self.n_components,
            n_contexts=self.n_contexts
        )

        self.epsilon, self._lambda = initialize_variational_parameters(
            random_state=self.random_state, dtype=self.dtype,
            n_components=self.n_components, n_contexts=self.n_contexts,
        )

        gamma, v = initialize_document_variational_parameters(
            random_state = self.random_state, dtype = self.dtype, 
            n_components = self.n_components,
            n_samples= len(freq_matrix),
        )

        freqs = self.get_freqs(freq_matrix)
        self.bounds = []

        _it = range(self.num_epochs)
        #if not self.quiet:
        #    _it = tqdm.tqdm(_it, desc = 'Training')

        corpus_size = len(freq_matrix)
        batch_size = min(self.batch_size, corpus_size)
        sstat_scale = corpus_size/batch_size
        batch_lda = batch_size == corpus_size

        logger.debug('Sstat scale: {}'.format(str(sstat_scale)))
        logger.debug('Batch size: {}'.format(str(batch_size)))

        if batch_lda:
            logger.warn('Running batch CTM algorithm')
        
        improvement_ema = None
        start_time = time.time()

        try:
            for epoch in _it:
                
                epoch_start_time = time.time()
                logger.debug('Begin Epoch {} E-step.'.format(str(epoch)))

                rho = 1. if batch_lda else self.get_rho(epoch)

                if not batch_lda:
                    logger.debug('\tRho: {:.2f}'.format(rho))

                if not batch_lda:
                    subsample = np.random.choice(corpus_size, 
                            size = batch_size, 
                            replace = False
                        )
                else:
                    subsample = np.arange(corpus_size)
                
                gamma[subsample], v[subsample], phi_matrix, weighted_phi = \
                    self._inference(
                        gamma = gamma[subsample], 
                        v = v[subsample], 
                        freq_matrix = freq_matrix[subsample], 
                        freqs = freqs[subsample],
                        difference_tol = self.difference_tol,
                        iterations = self.estep_iterations,
                    )

                if ~np.all(np.isfinite(weighted_phi)):
                    raise ValueError('Non-finite values detected in sufficient statistics. Stopping training to preserve current state.')

                logger.debug('\tBegin M-step.'.format(str(epoch)))

                self.epsilon, self._lambda = \
                    self._estimate_global_parameters(
                        weighted_phi, 
                        rho, 
                        sstat_scale = sstat_scale,
                    )

                if epoch > 0 and epoch % self.prior_update_every == 0:
                    
                    logger.debug('\tUpdating priors.')
                     # update local priors
                    self.mu, self.sigma = self.M_step_mu_sigma(
                        gamma[subsample], v[subsample], rho
                    )

                    self.b, self.nu = self._estimate_global_priors(
                            rho, optimize=batch_lda
                        )


                if epoch % self.eval_every == 0:
                   
                    if not batch_lda:
                        phi_matrix, weighted_phi = E_step_phi(
                            gamma, 
                            np.exp(self.logE_lambda)[:,:,:,None] * np.exp(self.logE_epsilon) , 
                            freq_matrix
                        )

                    self.bounds.append(
                        self._bound(gamma = gamma, v = v, phi_matrix = phi_matrix,
                            weighted_phi=weighted_phi, freqs=freqs)
                    )

                    if not np.isfinite(self.bounds[-1]):
                        logger.warn('\tBound is not finite on training data, stopping training.')
                        break

                    elif len(self.bounds) > 1:

                        improvement = self.bounds[-1] - self.bounds[-2]
                        logger.debug('\tBounds improvement: {:.2f}'.format(improvement))

                        improvement_ema = (1-0.1) * improvement_ema + 0.1*improvement
                        logger.debug('\tImprovement EMA: {:.2f}'.format(-improvement_ema))
                        if 0 < -improvement_ema < self.bound_tol:
                            break

                    else:
                        improvement_ema = self.bounds[-1]
                
                logger.debug('\tEpoch time: {:.2f} sec, total elapsed time {:.1f} min'\
                    .format(
                        time.time() - epoch_start_time,
                        (time.time() - start_time)/60
                    ))

        except KeyboardInterrupt:
            pass
    
        return self


    @staticmethod
    def _monte_carlo_softmax_expecation(gamma, v, n_samples = 300):

        def softmax(x):
            return np.exp(x)/np.exp(x).sum(axis = -1, keepdims = True)

        z = np.random.randn(n_samples, *gamma.shape)

        return softmax(v[None, :, :]*z + gamma[None,:,:]).mean(axis = 0)


    def _infer_document_variables(self, freq_matrix):

        gamma, v = initialize_document_variational_parameters(
                    random_state=self.random_state,
                    dtype=self.dtype,
                    n_samples=len(freq_matrix),
                    n_components=self.n_components
                )

        freqs = self.get_freqs(freq_matrix)

        return self._inference(
                gamma = gamma, v = v,
                freq_matrix = freq_matrix,
                freqs = freqs,
                difference_tol = 1e-3, 
                iterations = 10000,
                quiet=False,
            )

    @extract_freqmatrix
    def predict(self, freq_matrix):

        gamma, v, _, _ = self._infer_document_variables(freq_matrix)
        return self._monte_carlo_softmax_expecation(gamma, v)


    @extract_freqmatrix
    def _get_local_variational_parameters(self, freq_matrix):

        gamma, v, _, _ = self._infer_document_variables(freq_matrix)
        return self._monte_carlo_softmax_expecation(gamma, v)
        

    @extract_freqmatrix
    def score(self, freq_matrix):

        gamma, v, phi_matrix, weighted_phi = \
            self._infer_document_variables(freq_matrix)
        
        freqs = self.get_freqs(freq_matrix)

        return self._bound(
            gamma = gamma, v = v, phi_matrix=phi_matrix,
            weighted_phi=weighted_phi, freqs = freqs,
        )/np.sum(freq_matrix)
