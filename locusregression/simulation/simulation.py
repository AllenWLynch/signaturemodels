import numpy as np
from locusregression.corpus.readers import Sample
from locusregression.corpus.featurization import COSMIC_SORT_ORDER
from locusregression.corpus.corpus import Corpus, InMemorySamples
from locusregression.corpus.featurization import CONTEXT_IDX, MUTATIONS_IDX
import json
import os
import tqdm

TRANSITION_MATRIX = np.array([
    [0.99, 0.005, 0.005],
    [0.0, 0.97, 0.03],
    [0.015, 0.015, 0.97]
])

BETA_MATRIX = np.array([
    [1,-0.5,-0.5],
    [-0.5,1,-0.5],
    [-0.5,-0.5,1],
    [1,-1,1],
])

TRINUC_PRIORS = np.array([
    np.ones(32) * 5.,
    np.ones(32) * 20,
    np.ones(32) * 10,
])

SIGNAL_MEANS = np.array([1.,1.,1.])
SIGNAL_STD = np.array([0.3, 0.25, 0.5])


with open(os.path.join(os.path.dirname(__file__), 'cosmic.json'),'r') as f:
    COSMIC_SIGS = json.load(f)


class SimulatedCorpus:

    @staticmethod
    def cosmic_sig_to_matrix(cosmic_sig):
        
        sigmatrix = np.zeros((32,3))

        for key, p in cosmic_sig.items():
            'A[T>C]A'
            context = key[0] + key[2] + key[6]
            mutation = key[4]

            sigmatrix[CONTEXT_IDX[context], MUTATIONS_IDX[context][mutation]] += p

        return sigmatrix


    @staticmethod
    def create(
        seed = 0,
        n_cells = 100,
        log_mean_mutations = 5,
        log_std_mutations = 1.,
        pi_prior = 1.,
        n_loci = 1000,
        mutation_rate_noise = 0.1,
        state_transition_matrix = TRANSITION_MATRIX.copy(),
        beta_matrix = BETA_MATRIX.copy(),
        rate_function = None,
        trinucleotide_priors = TRINUC_PRIORS.copy(),
        signal_means = SIGNAL_MEANS.copy(),
        signal_std = SIGNAL_STD.copy(),
        exposures = None,*,
        cosmic_sigs,
    ):

        signatures = np.vstack([
            np.expand_dims(SimulatedCorpus.cosmic_sig_to_matrix(COSMIC_SIGS[sig_id]), 0)
            for sig_id in cosmic_sigs
        ])

        num_states = state_transition_matrix.shape[0]
        assert state_transition_matrix.shape == (num_states, num_states)

        num_signatures = signatures.shape[0]
        assert beta_matrix is None or beta_matrix.shape == (num_signatures, num_states)

        if beta_matrix is None:
            assert not rate_function is None

        assert signatures.shape == (num_signatures, 32, 3)

        assert isinstance(signal_means, np.ndarray) and isinstance(signal_std, np.ndarray) and \
            signal_means.shape == (num_states,) and signal_std.shape == (num_states,)

        assert trinucleotide_priors.shape == (num_states, 32)

        #assert isinstance(pi_prior, (int, float))

        shared_exposures = True
        if exposures is None:
            exposures = np.ones((n_cells, n_loci))
        else:
            shared_exposures = False
            assert exposures.shape == (n_cells, n_loci)

        randomstate = np.random.RandomState(seed)

        states = SimulatedCorpus.get_genomic_states(randomstate, 
            n_loci=n_loci, transition_matrix=state_transition_matrix)

        signals = SimulatedCorpus.get_signals(randomstate, state = states, 
            signal_means=signal_means, signal_std=signal_std)

        trinuc_distributions = np.vstack([
            randomstate.dirichlet(trinucleotide_priors[state])[None,:]
            for state in states
        ]).T

        cell_pi = randomstate.dirichlet(np.ones(num_signatures) * pi_prior, size = n_cells)
        cell_n_mutations = randomstate.lognormal(log_mean_mutations, log_std_mutations, size = n_cells).astype(int)

        samples = []
        for i, (pi, n_mutations, exposure) in tqdm.tqdm(enumerate(zip(cell_pi, cell_n_mutations, exposures)),
            total = len(cell_pi), ncols = 100, desc = 'Generating samples'):

            if i == 0 or not shared_exposures:
                psi_matrix = SimulatedCorpus.get_psi_matrix(signals, exposure, 
                        beta_matrix = beta_matrix, 
                        rate_function = rate_function,
                        mutation_rate_noise = mutation_rate_noise,
                        random_state= randomstate,
                )

            samples.append(
                SimulatedCorpus.simulate_sample(
                    randomstate, 
                    psi_matrix=psi_matrix,
                    trinuc_distributions=trinuc_distributions,
                    signatures=signatures,
                    pi = pi, 
                    n_mutations= n_mutations,
                    exposures = exposure[None,:],
                )
            )
        
        signals = np.vstack([
            signals, np.ones((1,signals.shape[-1]))
        ])

        corpus = Corpus(
                samples = InMemorySamples(samples),
                X_matrix = signals,
                trinuc_distributions = trinuc_distributions,
                feature_names = [f'Signal {i}' for i in range(num_states)] + ['Constant'],
                shared_correlates = shared_exposures,
        )

        generative_parameters = {
            'states' : states,
            'compositions' : cell_pi,
            'beta' : beta_matrix,
            'signatures' : signatures,
            'n_cells' : n_cells,
            'log_mean_mutations' : log_mean_mutations,
            'log_std_mutations' : log_std_mutations,
            'n_loci' : n_loci,
            'pi_prior' : pi_prior,
            'mutation_rate_noise' : mutation_rate_noise,
            'seed' : seed,
            'cosmic_sigs' : cosmic_sigs
        }

        return corpus, generative_parameters


    @staticmethod
    def get_genomic_states(randomstate,*, n_loci, transition_matrix):
        
        state = 0
        history = []
        for _ in range(n_loci):

            state = randomstate.choice(
                len(transition_matrix[state]),
                p = transition_matrix[state]
            )

            history.append(state)
            
        return np.array(history)


    @staticmethod
    def get_signals(randomstate,*, state, signal_means, signal_std):
        
        n_loci, n_features = len(state), len(signal_means)
        state_as_onehot = np.zeros(( n_loci, n_features ))
        state_as_onehot[ np.arange(n_loci), state] = np.ones(n_loci).astype(int)

        signals = signal_means[:,None]*state_as_onehot.T \
                + signal_std[:,None]*randomstate.randn(len(signal_means), len(state)) # S, L

        standard_signals = (signals - signals.mean(-1, keepdims = True))/signals.std(-1, keepdims = True)

        return standard_signals


    @staticmethod
    def get_psi_matrix(signals, exposure,
                        beta_matrix = None, 
                        rate_function = None,
                        mutation_rate_noise = 0.5,*,
                        random_state):
        
        if rate_function is None:
            assert not beta_matrix is None
            rate_function = lambda x : beta_matrix @ x

        psi_hat = exposure*np.exp(rate_function(signals) + mutation_rate_noise*random_state.randn(signals.shape[-1]))
        psi = psi_hat/psi_hat.sum(-1, keepdims = True) # K,L, sum l=1 -> L {psi_kl} = 1

        return psi


    @staticmethod
    def simulate_sample(randomstate,*,
            psi_matrix, trinuc_distributions, signatures,
            pi, n_mutations, exposures,
            ):

        loci, contexts, mutations = [],[],[]
        for i in range(n_mutations):

            sig = randomstate.choice(len(pi), p = pi)

            locus = randomstate.choice(psi_matrix.shape[1], p = psi_matrix[sig,:])

            context_dist = signatures[sig].sum(-1)
            
            context = randomstate.choice(
                32, p = context_dist*trinuc_distributions[:,locus] * (1/np.sum(context_dist*trinuc_distributions[:,locus]))
            )

            mutation_dist = signatures[sig, context]/signatures[sig, context].sum()
            mutation = randomstate.choice(3, p = mutation_dist)

            loci.append(locus)
            contexts.append(context)
            mutations.append(mutation)

        return {
            **Sample._aggregate_counts(mutations, contexts, loci), 
            'window_size' : exposures
            }


    def from_model(
        model, 
        corpus,
        n_cells = 100,
        seed = 0,
        ):

        randomstate = np.random.RandomState(seed)
        
        num_signatures = model.n_components
        pi_prior = model.alpha
        n_loci = model.n_loci
        beta_matrix = model.beta_mu
        signals = corpus.X_matrix
        trinuc_distributions = corpus.trinuc_distributions

        signatures = np.vstack([
            np.expand_dims(SimulatedCorpus.cosmic_sig_to_matrix(dict(zip(COSMIC_SORT_ORDER, model.signature(k)))), 0)
            for k in range(model.n_components)
        ])

        cell_pi = randomstate.dirichlet(np.ones(num_signatures) * pi_prior, size = n_cells)
        cell_n_mutations = [sum(sample['count']) for sample in corpus]

        psi_matrix = SimulatedCorpus.get_psi_matrix(beta_matrix, signals)

        exposures = np.ones((1, n_loci))

        samples = []
        for pi, n_mutations in tqdm.tqdm(zip(cell_pi, cell_n_mutations),
            total = len(cell_pi), ncols = 100, desc = 'Generating samples'):

            samples.append(
                SimulatedCorpus.simulate_sample(
                    randomstate, 
                    psi_matrix=psi_matrix,
                    trinuc_distributions=trinuc_distributions,
                    signatures=signatures,
                    pi = pi, 
                    n_mutations= n_mutations,
                    exposures = exposures.copy(),
                )
            )
        
        return Corpus(
                samples = InMemorySamples(samples),
                X_matrix = signals,
                trinuc_distributions = trinuc_distributions,
                feature_names = corpus.feature_names,
                shared_correlates = True,
        )