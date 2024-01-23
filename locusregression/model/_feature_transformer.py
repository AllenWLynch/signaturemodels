import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, \
    QuantileTransformer, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone, BaseEstimator
import logging
from numpy import array
logger = logging.getLogger(' LocusRegressor')

class ClassStratifiedTransformer(BaseEstimator):

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.transformer.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X, y)



class FeatureTransformer:

    def __init__(self, categorical_encoder = OneHotEncoder(sparse_output=False, drop='first')):
        self.categorical_encoder = clone(categorical_encoder)


    def _assemble_matrix(self, corpus_states):
        
        feature_matrix = pd.concat([
            pd.DataFrame(
                {feature_name : state.features[feature_name]['values'] for feature_name in self.feature_names_}
            )
            for state in corpus_states.values()
        ])

        corpus_labels = array([
            name
            for name, state in corpus_states.items()
            for _ in  range(len(state.features[self.feature_names_[0]]['values']))
        ]).astype(str)

        return feature_matrix, corpus_labels



    def list_feature_groups(self):
        
        feature_groups_dict_ = defaultdict(list)
        for feature_name, groups in zip(
            self.feature_names_, self.groups_
        ):
            for group in groups:
                try:
                    feature_groups_dict_[group].append(
                        self.feature_names_out.index(feature_name)
                    )
                except ValueError:
                    feature_groups_dict_[group].extend(
                        [idx for idx, name in enumerate(self.feature_names_out) if name.startswith(feature_name + '_')]
                    )

        return list(feature_groups_dict_.values())


    @property
    def feature_names_out(self):
        return list(self.transformer_.get_feature_names_out())
    

    def list_categorical_features(self):

        encoder_name = type(self.categorical_encoder).__name__.lower()
        _slice = self.transformer_.output_indices_[encoder_name]
        
        return list(range(_slice.start, _slice.stop))


    def fit(self, corpus_states):
        
        example_features = next(iter(corpus_states.values())).features
        
        self.feature_names_ = list(example_features.keys())
        self.feature_types_ = [example_features[feature]['type'] for feature in self.feature_names_]
        self.groups_ = [example_features[feature]['group'].split(',') for feature in self.feature_names_]
        
        feature_type_dict_ = defaultdict(list)        
        for idx, feature_type in enumerate(self.feature_types_):
            feature_type_dict_[feature_type].append(idx)

        self.transformer_ = ColumnTransformer(
            ('power', ClassStratifiedTransformer(PowerTransformer()), feature_type_dict_['power']),
            ('minmax', ClassStratifiedTransformer(MinMaxScaler()), feature_type_dict_['minmax']),
            ('quantile', ClassStratifiedTransformer(QuantileTransformer(output_distribution='uniform')), feature_type_dict_['quantile']),
            ('standardize', ClassStratifiedTransformer(StandardScaler()), feature_type_dict_['standardize']),
            ('robust', ClassStratifiedTransformer(RobustScaler()), feature_type_dict_['robust']),
            ('categorical', self.categorical_encoder, feature_type_dict_['categorical']),
            remainder='passthrough',
            verbose_feature_names_out=False,
        )

        matrix, labels = self._assemble_matrix(corpus_states)
        self.transformer_.fit(matrix, y=labels)

        if len(feature_type_dict_['categorical']) > 0:
            logger.info(
                f'Found categorical features: {", ".join(self.feature_names_out[i] for i in self.list_categorical_features())}'
            )
            
        for feature_group in self.list_feature_groups():
            if len(feature_group) > 0:
                logger.info(
                    f'Found feature group: {", ".join(self.feature_names_out[i] for i in feature_group)}'
                )

        return self


    def transform(self, corpus_states):

        for corpus_state in corpus_states.values():
            assert corpus_state.feature_names == self.feature_names_

        matrix, labels = self._assemble_matrix(corpus_states)
        transformed_matrix = self.transformer_.transform(matrix, y=labels)

        return transformed_matrix
    
