
import numpy as np
from ._dirichlet_update import update_alpha
from ..simulation import SimulatedCorpus, COSMIC_SIGS
from sklearn.linear_model import PoissonRegressor
from scipy.special import logsumexp
from sklearn.preprocessing import OneHotEncoder
from ._feature_transformer import FeatureTransformer, CardinalityTransformer
from functools import reduce
from pandas import DataFrame

def _get_linear_model(*args, **kw):
    return PoissonRegressor(
        alpha = 0, 
        solver = 'newton-cholesky',
        warm_start = True,
        fit_intercept = False,
    )


class DummyCorpus:

    def __init__(self, corpus):
        self.context_frequencies = corpus.context_frequencies
        self.shared_correlates = corpus.shared_correlates


class ModelState:

    def __init__(self,
                fix_signatures = None,
                pseudocounts = 10000,
                get_model_fn = _get_linear_model,
                categorical_encoder = OneHotEncoder(sparse_output=False, drop='first'),
                signature_reg = 0.,*,
                corpus_states,
                n_components,
                random_state, 
                empirical_bayes,
                genome_context_frequencies,
                feature_dim,
                cardinality_features_dim,
                cardinalities_dim,
                locus_dim,
                context_dim,
                mutation_dim,
                attribute_dim,
                dtype,
                **kw,
            ):
        
        assert isinstance(n_components, int) and n_components >= 1
        self.n_components = n_components
        self.n_loci = locus_dim
        self.cardinality_features_dim = cardinality_features_dim
        self.random_state = random_state
        self.empirical_bayes = empirical_bayes
        self.n_contexts = context_dim

        self._lambda = self.random_state.gamma(100, 1/100, 
                                               (n_components, context_dim),
                                              ).astype(dtype, copy=False)
        
        self._rho = self.random_state.gamma(100, 1/100,
                                               (n_components, context_dim, mutation_dim),
                                              ).astype(dtype, copy = False)
        
        self._tau = np.ones((n_components, cardinality_features_dim))\
            .astype(dtype, copy=False)
        
        # placeholder for when attributes come into play
        #self.psi
        
        # this will need to be generalized
        if not fix_signatures is None:
            self._fix_signatures(fix_signatures,
                                 n_components = n_components,
                                 genome_context_frequencies = genome_context_frequencies,
                                 pseudocounts = pseudocounts
                                )
        else:
            self.fixed_signatures = [False]*n_components
        
        self.feature_transformer = FeatureTransformer(
                                        categorical_encoder=categorical_encoder,
                                    ).fit(corpus_states)

        self._fit_corpus_encoder(corpus_states)

        design_matrix = self._get_design_matrix(corpus_states)
        X = self.feature_transformer.transform(corpus_states)

        self.rate_models = [
            get_model_fn(
                design_matrix = design_matrix, 
                features = X,
                categorical_features = self.feature_transformer.list_categorical_features(),
                interaction_groups = self.feature_transformer.list_feature_groups(),
            ) 
            for _ in range(n_components)
        ]

        self.n_distributions = design_matrix.shape[1]

        self.context_models = [
            PoissonRegressor(alpha = signature_reg, 
                             fit_intercept=False, 
                             warm_start=True) #, solver='newton-cholesky')
            for _ in range(n_components)
        ]

        self.cardinality_models = [
            PoissonRegressor(alpha = 0.,
                             fit_intercept=False,
                                warm_start=True)
            for _ in range(n_components)
        ]

        if cardinality_features_dim > 0:
            self.strand_transformer = CardinalityTransformer().fit(corpus_states)
        else:
            self.strand_transformer = None


    @property
    def lambda_(self):
        return self._lambda
    
    @property
    def rho_(self):
        return self._rho
    
    @property
    def tau_(self):
        return self._tau


    def _fix_signatures(self, fix_signatures,*,
                        n_components, 
                        genome_context_frequencies, 
                        pseudocounts = 10000):
            
        assert isinstance(fix_signatures, list) and len(fix_signatures) <= n_components, \
                'fix_signatures must be a list of signature names with a most n_components elements'
        
        self.fixed_signatures = [True]*len(fix_signatures) + [False]*(n_components - len(fix_signatures))

        for i, sig in enumerate(fix_signatures):
            
            try:
                COSMIC_SIGS[sig]
            except KeyError:
                raise ValueError(f'Unknown signature {sig}')
            
            sigmatrix = SimulatedCorpus.cosmic_sig_to_matrix(COSMIC_SIGS[sig])
            
            self._rho[i] = sigmatrix * pseudocounts + 1.
            self._lambda[i] = sigmatrix.sum(axis = -1) * pseudocounts/genome_context_frequencies + 1.


    def _fit_corpus_encoder(self, corpus_states):

        corpus_names = list(corpus_states.keys())

        self.corpus_intercept_encoder_ = OneHotEncoder(
                        sparse_output=True,
                        drop = None,
                    ).fit(
                        np.array(corpus_names).reshape((-1,1))
                    )
    

    def _get_onehot_column(self, corpus_states, n_repeats):
        labels = np.concatenate([[name]*n_repeats for name in corpus_states.keys()])
        # One-hot encode the labels
        encoded_labels = self.corpus_intercept_encoder_.transform(
            labels.reshape((-1,1))
        )
        return encoded_labels


    def _get_design_matrix(self, corpus_states):
        n_loci = next(iter(corpus_states.values())).n_loci
        return self._get_onehot_column(corpus_states, n_loci)
    

    
    @staticmethod
    def _svi_update_fn(old_value, new_value, learning_rate):
        return (1-learning_rate)*old_value + learning_rate*new_value
    

    def _svi_update(self, param, new_value, learning_rate):
        
        self.__setattr__(
            param, self._svi_update_fn(self.__getattribute__(param), new_value, learning_rate)
        )

        return self.__getattribute__(param)


    def update_rho(self, sstats, corpus_states, learning_rate):
        
        new_rho = np.vstack([
            np.expand_dims(sstats.mutation_sstats[k], axis = 0)
            for k in range(self.n_components)
        ])

        new_rho = new_rho/np.sum(new_rho, axis = -1, keepdims = True)

        self._svi_update('_rho', new_rho, learning_rate)



    def _get_tau_features(self, corpus_states):
        
        n_bins = next(iter(corpus_states.values())).n_loci

        def _get_cardinality_features(corpus_state):
            strand_features = self.strand_transformer.transform({corpus_state.name : corpus_state})
            return np.concatenate([strand_features, -1*strand_features], axis=0)
        
        X = np.concatenate(
            [
                _get_cardinality_features(state)
                for state in corpus_states.values()
            ], axis = 0
        )
        
        X = np.hstack([
            X, self._get_onehot_column(corpus_states, 2*n_bins).toarray()
        ])

        x_cols = ['X' + str(i) for i in range(X.shape[1])]
        df = DataFrame(X, columns = x_cols)
        return df


    def _get_tau_targets(self, k, sstats, corpus_states):

        n_bins = next(iter(corpus_states.values())).n_loci

        def _get_cardinality_exposure(corpus_state):
            # 1xCx1 @ DxCxL --> DxL + 1xL --> DxL --> [L_d1 \+ L_d2]
            return (
                (self.lambda_[k][None, :, None] * corpus_state.context_frequencies).sum(1) * \
                corpus_state.exposures * np.exp(corpus_state.theta_[k])[None,:]
            ).ravel()
        
        eta = np.concatenate([_get_cardinality_exposure(state) for state in corpus_states.values()]) # I x C -> I*C
        target = np.concatenate([sstats[name].tau_sstats(k, n_bins).ravel() for name in corpus_states.keys()])

        m = (target/eta).mean()
        sample_weights = eta * m

        return target, sample_weights
    
    
    def _tau_update(self, sstats, corpus_states):

        df = self._get_tau_features(corpus_states)
        x_cols = list(df.columns.values)

        for k in range(self.n_components):
            df['target' + str(k)], df['weight' + str(k)] = self._get_tau_targets(k, sstats, corpus_states)

        df = df.groupby(x_cols).sum().reset_index()

        return np.array([
            np.exp(
                self.cardinality_models[k]\
                .fit(
                    df[x_cols].values,
                    df['target' + str(k)]/df['weight' + str(k)],
                    sample_weight=df['weight' + str(k)]/df['weight' + str(k)].mean()
                ).coef_[:self.cardinality_features_dim]
            )
            for k in range(self.n_components)
        ])

    
    def update_tau(self, sstats, corpus_states, learning_rate):
        _tau = self._tau_update(sstats, corpus_states)
        print(_tau.ravel())
        self._svi_update('_tau', _tau, learning_rate)


    def _lambda_update(self, k, sstats, corpus_states):

        def _get_context_exposure(corpus_state):
            # C x L @ L -> D x C --> C
            return (
                (np.exp(corpus_state._get_log_strand_effects(k, self))*corpus_state.context_frequencies).sum(0) @ \
                (corpus_state.exposures.ravel() * np.exp(corpus_state.theta_[k]))
            )
        
        I = len(corpus_states.keys())
        eta = np.concatenate([_get_context_exposure(state) for state in corpus_states.values()]) # I x C -> I*C
        target = np.concatenate([sstats[name].lambda_sstats(k) for name in corpus_states.keys()])

        m = (target/eta).mean()
        sample_weights = eta * m

        # remove any samples with zero weight to avoid divide-by-zero errors
        zero_mask = sample_weights == 0

        if (target[zero_mask] > 0).any():
            raise ValueError('A sample weight is zero but the target is positive')
        else:
            target = target[~zero_mask]
            sample_weights = sample_weights[~zero_mask]

        X = np.hstack([
                np.tile(
                    np.diag(np.ones(self.n_contexts)),
                    (I, 1)
                ),
                self._get_onehot_column(corpus_states, self.n_contexts).toarray()
            ])

        
        return np.exp(
            self.context_models[k]\
            .fit(
                X, 
                target/sample_weights,
                sample_weight=sample_weights/sample_weights.mean()
            ).coef_[:self.n_contexts]
        )

   
    def update_lambda(self, sstats, corpus_states, learning_rate):
        
        _lambda = np.array([
            self._lambda_update(k, sstats, corpus_states)
            for k in range(self.n_components)
        ])

        self._svi_update('_lambda', _lambda, learning_rate)
    

    def _get_targets(self, sstats, corpus_states):
        
        n_bins = next(iter(corpus_states.values())).n_loci

        exposures = np.concatenate(
            [state.exposures for state in corpus_states.values()],
            axis = 0,
        )

        for k in range(self.n_components):

            current_lograte_prediction = np.array(
                [state.theta_[k] for state in corpus_states.values()]
            ).ravel()

            context_effect = np.array(
                [state.signature_effects_[k].sum(axis = (0,1)) for state in corpus_states.values()]
            ).ravel()

            target = np.concatenate([sstats[name].theta_sstats(k, n_bins) for name in corpus_states.keys()])
            eta = (exposures * context_effect).ravel()

            # rescale the targets to mean 1 so that the learning rate is comparable across components and over epochs
            m = (target/eta).mean()
            sample_weights = eta * m

            # remove any samples with zero weight to avoid divide-by-zero errors
            zero_mask = sample_weights == 0

            if (target[zero_mask] > 0).any():
                raise ValueError('A sample weight is zero but the target is positive')
            else:
                target = target[~zero_mask]
                sample_weights = sample_weights[~zero_mask]
                current_lograte_prediction = current_lograte_prediction[~zero_mask]

            y_tild = target/sample_weights

            yield (
                y_tild,
                sample_weights/sample_weights.mean(), # rescale the weights to mean 1 so that the learning rate is comparable across components and over epochs
                current_lograte_prediction
            )


    def update_rate_model(self, sstats, corpus_states, learning_rate):

        design_matrix = self._get_design_matrix(corpus_states)
        X = self.feature_transformer.transform(corpus_states)

        X = np.hstack([np.nan_to_num(X, nan=0), design_matrix.toarray()])

        for k, (y, sample_weights, lograte_prediction) in enumerate(
            self._get_targets(sstats, corpus_states)
        ):
            
            # store the current model state (ignore the intercept fits)
            try:
                old_coef = self.rate_models[k].coef_.copy()
            except AttributeError:
                old_coef = np.zeros(X.shape[1])            

            # update the model with the new suffstats
            self.rate_models[k].fit(
                X, 
                y,
                sample_weight=sample_weights,
            )

            # merge the new model state with the old
            self.rate_models[k].coef_ = self._svi_update_fn(
                old_coef, 
                self.rate_models[k].coef_, 
                learning_rate
            )

    def update_state(self, sstats, corpus_states, learning_rate):
        
        update_params = ['rate_model','lambda','rho', 'tau']
        
        for param in update_params:
            self.__getattribute__('update_' + param)(sstats, corpus_states, learning_rate) # call update function

        

class CorpusState(ModelState):
    '''
    Holds corpus-level parameters, like the current mutation rate estimates and 
    corpus-specific priors over signatures
    '''


    def __init__(self, corpus,*,pi_prior,n_components, dtype, random_state,
                 subset_sample = 1):

        self.corpus = corpus
        self.random_state = random_state
        self.n_components = n_components
        self.dtype = dtype
        self.pi_prior = pi_prior
        self.n_loci = corpus.locus_dim
        self.subset_sample = subset_sample
        
        self.n_samples = len(corpus)
        
        self.alpha = np.ones(self.n_components)\
            .astype(self.dtype, copy=False)*self.pi_prior
        
        self._theta = self._get_baseline_prediction(
            self.n_components, self.n_loci, self.dtype
        )

        
    def _get_baseline_prediction(self, n_components, n_loci, dtype):
        return np.zeros((n_components, n_loci), dtype = dtype)
    

    def clone_corpusstate(self, corpus):
        
        new_state = self.__class__(
            corpus = corpus,
            pi_prior= self.pi_prior,
            n_components=self.n_components,
            dtype = self.dtype,
            random_state = self.random_state,
            subset_sample=self.subset_sample
        )
        new_state.alpha = self.alpha.copy()

        return new_state
    

    def subset_corpusstate(self, corpus, locus_subset):

        newstate = self.__class__(
            corpus = corpus,
            pi_prior= self.pi_prior,
            n_components=self.n_components,
            dtype = self.dtype,
            random_state = self.random_state,
            subset_sample=len(locus_subset)/self.n_loci
        )

        newstate.alpha = self.alpha.copy()
        newstate._theta = self.theta_[:, locus_subset]
        newstate._log_denom = self.log_denom_

        return newstate
    

    def _get_log_strand_effects(self, k, model_state):

        if self.corpus.cardinality_features_dim == 0:
            return np.zeros((2, 1, self.n_loci)).astype(self.dtype, copy=False)
        
        strand_features = model_state.strand_transformer.transform(
                                    {self.name : self}
                                )
        if strand_features.ndim == 1:
            strand_features = np.expand_dims(strand_features, axis=1)

        strand_factors = strand_features @ np.log(model_state.tau_[k])

        # 2 x L --> 2 x 1 x L
        strand_effects = np.expand_dims( np.array([strand_factors, -strand_factors]), axis = 1)

        return strand_effects
    
    
    def _get_log_signature_effect(self, k, model_state):

        # (2 x 1 x L) + (2 x C x L) + (1 x C x 1) %R% -> (2C x L)
        return (
            self._get_log_strand_effects(k, model_state) \
            + np.log(self.context_frequencies) \
            + np.log(model_state.lambda_[k][None,:,None])
        )
                
    
    def _get_log_component_mutation_rate(self, k, model_state, exposures):
        
        # Cx1 + CxL -> CxL
        return np.nan_to_num(
            np.log(self.signature_effects_[k]) + \
            + self.theta_[k][None,None,:] \
            + np.log(exposures)[None, :, :] \
            - self.log_denom_[k],
            nan = -np.inf
        )
    

    def _get_log_marginal_effect_rate(self, pi, model_state, exposures):

        return np.log(
            reduce(
                lambda x, k : x + ( pi[k]*np.exp(self._get_log_component_mutation_rate(k, model_state, exposures)) ),
                range(self.n_components),
                np.zeros_like(self.context_frequencies)
            )
        )

    
    def get_log_component_effect_rate(self, model_state, exposures, use_context=True):
        '''
        Returns a (Z x C x L) tensor of the log of the component-wise mutation rate effects
        '''
        return np.array([
            np.nan_to_num(self._get_log_component_mutation_rate(k, model_state, exposures), nan = -np.inf)
            for k in range(self.n_components)
        ])


    def _get_log_denom(self):
        # (KxC) @ (CxL) |-> (KxL)
        # K x 2 x C x L -> K x L
        logits = np.log(self.signature_effects_.sum(axis = (1,2))) + self.theta_ + np.log(self.exposures)
        return logsumexp(logits, axis = 1, keepdims = True)

    
    @property
    def signature_effects_(self):
        return self._signature_effects


    def _update_stored_params(self, model_state):
        
        self._signature_effects = np.array([
            np.exp(self._get_log_signature_effect(k, model_state))
            for k in range(self.n_components)
        ])

        self._log_denom = self._get_log_denom()
        
        return self
    

    def update(self, model_state, from_scratch=False):
        
        design_matrix = model_state._get_design_matrix({self.name : self})
        X = model_state.feature_transformer.transform(
                                    {self.name : self}
                                )

        X = np.hstack([np.nan_to_num(X, nan=0), design_matrix.toarray()])

        self._theta = np.array([
            np.log(model_state.rate_models[k].predict(X).T)
            for k in range(self.n_components)
        ])

        self._update_stored_params(model_state)

        return self

    
    def update_alpha(self, sstats, learning_rate):
        _alpha = update_alpha(self.alpha, sstats[self.corpus.name].alpha_sstats)
        self._svi_update('alpha', _alpha, learning_rate)


    def set_alpha(self, gammas):
        _alpha = update_alpha(self.alpha, gammas)
        self._svi_update('alpha', _alpha, 1)


    def update_gamma(self, sstats, learning_rate):
        _gamma = sstats.gamma_sstats[self.corpus.name]
        self._svi_update('gamma', _gamma, learning_rate)


    @property
    def theta_(self):
        return self._theta
    
    @property
    def log_denom_(self):
        return self._log_denom

    @property
    def exposures(self):
        assert self.corpus.shared_exposures
        return self.corpus.exposures        

    @property
    def context_frequencies(self):
        return self.corpus.context_frequencies
    
    @property
    def features(self):
        return self.corpus.features
    
    @property
    def name(self):
        return self.corpus.name
        
    @property
    def feature_names(self):
        return self.corpus.feature_names

    def as_dummy(self):
        self.corpus = DummyCorpus(self.corpus)
        return self
    