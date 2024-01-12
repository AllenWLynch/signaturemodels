import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.base import clone
import logging
logger = logging.getLogger(' LocusRegressor')


class FeatureTransformer:

    def __init__(self, categorical_encoder = OneHotEncoder(sparse=False, drop='first')):
        self.categorical_encoder = clone(categorical_encoder)


    def _assemble_matrix(self, corpus_states):
        return pd.concat([
            pd.DataFrame(
                {feature_name : state.features[feature_name]['values'] for feature_name in self.feature_names_}
            )
            for state in corpus_states.values()
        ])


    def list_feature_groups(self):
        return list(self.feature_groups_dict_.values())
    

    def list_categorical_features(self):
        return self.feature_type_dict_['categorical']


    def fit(self, corpus_states):
        
        example_features = next(iter(corpus_states.values())).features
        
        self.feature_names_ = list(example_features.keys())
        self.feature_types_ = [example_features[feature]['type'] for feature in self.feature_names_]
        self.groups_ = [example_features[feature]['group'] for feature in self.feature_names_]
        
        self.feature_type_dict_ = defaultdict(list)        
        for idx, feature_type in enumerate(self.feature_types_):
            self.feature_type_dict_[feature_type].append(idx)

        self.feature_groups_dict_ = defaultdict(list)
        for idx, feature_group in enumerate(self.groups_):
            self.feature_groups_dict_[feature_group].append(idx)

        matrix = self._assemble_matrix(corpus_states)

        self.transformer_ = make_column_transformer(
            (PowerTransformer(), self.feature_type_dict_['continuous']),
            (self.categorical_encoder, self.feature_type_dict_['categorical']),
            (MinMaxScaler(), self.feature_type_dict_['distance']),
            remainder='passthrough',
        )

        self.transformer_.fit(matrix)

        if len(self.feature_groups_dict_['categorical']) > 0:
            logger.info(
                f'Found categorical features: {", ".join(self.feature_names_[i] for i in self.list_categorical_features())}'
            )
            
        for feature_group in self.list_feature_groups():
            if len(feature_group) > 0:
                logger.info(
                    f'Found feature group: {", ".join(self.feature_names_[i] for i in feature_group)}'
                )
            
        
        return self


    def transform(self, corpus_states):

        for corpus_state in corpus_states.values():
            assert corpus_state.feature_names == self.feature_names_

        matrix = self._assemble_matrix(corpus_states)
        transformed_matrix = self.transformer_.transform(matrix)

        return transformed_matrix
    
