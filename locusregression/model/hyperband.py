import tqdm
import time
import numpy as np
from math import log, floor, ceil
import tqdm
import multiprocess as mp
import logging


class Bracket:

    def __init__(self,
            max_candidates = np.inf,*,
            budget,
            max_resources,
            factor,
            bracket_num,
            random_model_function,
            randomstate,
        ):

        self.bracket_num = bracket_num
        self.iteration = 0
        self.next_generation = []
        self.finished = False
        self.factor = factor
        
        self.n_configs = min(
            ceil(budget/max_resources * (factor**bracket_num)/(bracket_num + 1)),
            max_candidates
        )

        self.base_resouces = max_resources*factor**(-self.bracket_num)
        
        self.models = [random_model_function(randomstate) for _ in range(self.n_configs)]

        
    def __len__(self):
        return sum([self.get_n(i)*self.get_r(i)
                    for i in range(self.bracket_num + 1)])
    
    
    def get_n(self, i):
        return max( floor(self.n_configs*self.factor**(-i)), 1 )
    

    def get_r(self,i):
        return ceil(self.base_resouces*self.factor**i)
    
    @property
    def n_i(self):
        return self.get_n(self.iteration)

    @property
    def r_i(self):
        return self.get_r(self.iteration)


    def has_next(self):
        return not self.finished


    def pop(self):

        next_model = self.models.pop()
        return next_model, self.r_i


    def cull(self, generation):

        top_n = floor(self.n_i/self.factor)

        models, losses = list(zip(*generation))
                
        ranks = np.argsort(-np.array(losses)).argsort()
        models = [model for rank, model in zip(ranks, models) if rank < top_n]
        
        return models


    def push(self, model, loss):

        self.next_generation.append(
            (model, loss)
        )

        if len(self.next_generation) == self.n_i:
            
            if self.iteration < self.bracket_num :
                
                self.models = self.cull(self.next_generation)
                self.iteration+=1

                if self.iteration == self.bracket_num:
                    self.finished = True
                else:
                    self.next_generation = []
    
    def __str__(self):
        return f'Bracket {self.bracket_num}: Evaluating {self.n_configs} models'


class HyperBand:

    def __init__(self,
        seed = 0,
        max_resources = 300,
        factor = 3,
        max_candidates = np.inf,*,
        random_model_function,
    ):

        self.factor = factor
        self.max_resources = max_resources

        self.randomstate = np.random.RandomState(seed)
    
        self.max_resources = max_resources
        self.s_max = floor( log(self.max_resources)/log(self.factor) )
        self.budget = (self.s_max + 1)*self.max_resources

        self.brackets = [
            Bracket(
                budget = self.budget,
                max_resources = self.max_resources,
                random_model_function = random_model_function,
                factor = self.factor,
                randomstate = self.randomstate,
                max_candidates = max_candidates,
                bracket_num=s
            ) for s in range(1, self.s_max)
        ]
        
        
    def __len__(self):
        return sum([len(bracket) for bracket in self.brackets])

    def __str__(self):
        return '\n'.join(
            [str(bracket) for bracket in self.brackets]
        )
    
    
    def pop(self):

        for bracket_num, bracket in enumerate(self.brackets):
            
            try:
                return (bracket_num, *bracket.pop())
            except IndexError:
                pass

        raise IndexError()


    def push(self, bracket_num, model, loss):
        
        self.brackets[bracket_num].push(model, loss)

        
    def has_next(self):
        return any([b.has_next() for b in self.brackets])

    def survivors(self):
        return [model for bracket in self.brackets for model in bracket.models]
    

def run_hyperband(
    random_model_func,
    eval_func,
    records_func,
    factor = 3,
    max_resources = 300,
    seed = 0,
    n_jobs = 1,
    max_candidates = np.inf,
):
    
    band = HyperBand(
        random_model_function = random_model_func,
        factor = factor,
        max_resources = max_resources,
        seed = seed,
        max_candidates = max_candidates,
    )

    print(f'Running HyperBand with {n_jobs} jobs.')
    print(band)

    def worker(input_queue, output_queue):

        while True:

            bracket, model, resources = input_queue.get(True)

            loss = eval_func(model, resources)

            output_queue.put(
                (bracket, model, loss, resources)
            )


    input_queue, output_queue = mp.Queue(), mp.Queue()
    records = []
    
    try:
        with tqdm.tqdm(total = len(band), desc = 'Evaluating model configurations', ncols = 100) as bar:
            with mp.Pool(n_jobs, worker, (input_queue, output_queue,) ) as pool:

                submitted_jobs = 0

                while band.has_next() \
                    or not input_queue.empty() or not output_queue.empty() \
                    or submitted_jobs > 0:

                    try:

                        bracket, model, resources = band.pop()
                        input_queue.put((bracket, model, resources))
                        submitted_jobs+=1

                    except IndexError:

                        if not output_queue.empty():

                            bracket, model, loss, resources = output_queue.get(1)
                            band.push(bracket, model, loss)

                            records.append(
                                records_func(bracket, model, loss, resources)
                            )

                            bar.update(resources)
                            submitted_jobs -= 1

                        else:
                            time.sleep(1)

            bar.update(len(band) - bar.n)
    
    except KeyboardInterrupt:
        pass
    
    return records