import numpy as np
import argparse
import pickle

transition_matrix = np.array([
    [0.99, 0.005, 0.005],
    [0.0, 0.97, 0.03],
    [0.015, 0.015, 0.97]
])

signal_means = np.array([1.,1.,1.])
signal_stds = np.array([0.3, 0.25, 0.5])

trinuc_priors = np.array([
    np.ones(32) * 20.,
    np.ones(32) * 20,
    np.ones(32) * 20,
])

def medium_config(*,
        n_cells,
        log_mean_mutations):


    beta_matrix = np.array([
        [1,-0.5,0],
        [-0.5,1,-0.5],
        [-0.5,-0.5,1],
        [1,-1,1],
        [0.,1.,1],
    ])


    kwargs = dict(
        state_transition_matrix = transition_matrix,
        beta_matrix = beta_matrix,
        trinucleotide_priors = trinuc_priors,
        signal_means= signal_means,
        signal_std = signal_stds,
        mutation_rate_noise = 0.1,
        pi_prior = 1,
        n_loci = 250000,
        cosmic_sigs=['SBS1','SBS8','SBS3','SBS5','SBS2'],
        seed = 10,
        n_cells = n_cells,
        log_mean_mutations = log_mean_mutations,
        log_std_mutations = 0.5, 
    )

    return kwargs


def easy_config(*,
        n_cells,
        log_mean_mutations):


    beta_matrix = np.array([
        [1,-0.5,0],
        [-0.5,1,-0.5],
        [-0.5,-0.5,1],
    ])

    kwargs = dict(
        state_transition_matrix = transition_matrix,
        beta_matrix = beta_matrix,
        trinucleotide_priors = trinuc_priors,
        signal_means= signal_means,
        signal_std = signal_stds,
        mutation_rate_noise = 0.1,
        pi_prior = 1,
        n_loci = 250000,
        cosmic_sigs=['SBS1','SBS8','SBS5'],
        seed = 10,
        n_cells = n_cells,
        log_mean_mutations = log_mean_mutations,
        log_std_mutations = 0.5, 
    )

    return kwargs


def hard_config(*,
        n_cells,
        log_mean_mutations):


    beta_matrix = np.array([
        [1,-0.5,0],
        [-0.5,1,-0.5],
        [-0.5,-0.5,1],
        [1,-1,1],
        [0.,1.,1],
        [0.,0.,0.,],
        [1,-0.5,0],
        [0.5,0.5,-1],
        [1,0.,0.5],
    ])


    kwargs = dict(
        state_transition_matrix = transition_matrix,
        beta_matrix = beta_matrix,
        trinucleotide_priors = trinuc_priors,
        signal_means= signal_means,
        signal_std = signal_stds,
        mutation_rate_noise = 0.1,
        pi_prior = 1,
        n_loci = 250000,
        cosmic_sigs=['SBS1','SBS8','SBS3','SBS5','SBS2','SBS93','SBS4','SBS7','SBS26'],
        seed = 10,
        n_cells = n_cells,
        log_mean_mutations = log_mean_mutations,
        log_std_mutations = 0.5, 
    )

    return kwargs

parser = argparse.ArgumentParser()
parser.add_argument('outdir')
parser.add_argument('test', choices=['easy','medium','hard'])

if __name__ == "__main__":

    args = parser.parse_args()

    tests = {
        'easy' : easy_config,
        'medium' : medium_config,
        'hard' : hard_config
    }

    i=0
    for n_cells in [1,25,50,100,200]:
        for log_mean_mutations in [3,4,5,6,7]:
            i+=1
            
            config = tests[args.test](n_cells=n_cells, log_mean_mutations=log_mean_mutations)

            with open(args.outdir + f'/{i}_config.pkl','wb') as f:
                pickle.dump(config, f)

    print('Created tests:\n' + ','.join(map(str, range(1,i))))
