from .corpus import Sample, BaseCorpus, Corpus, MixedCorpus
from .featurization import *
from .simulation import simulate_corpus

def load_corpus(corpus):
    return BaseCorpus.load(corpus)