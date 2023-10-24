from sklearn.preprocessing import PowerTransformer, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import argparse

epigenetic_features = \
'''DNAMethylSBS
H3K27ac
H3K4me1
H3K4me3
H3K9me3
H3K27me3
H3K27ac
H3K36me3
RNAseq'''\
    .split('\n')


def normalize_features(raw_features):

    X = pd.read_csv(raw_features)

    pipeline = Pipeline([
        ('epigenetics', ColumnTransformer(
            [
                ('power', PowerTransformer(), epigenetic_features),
                ('scaler', StandardScaler(), epigenetic_features),
                ('interactions', PolynomialFeatures(degree = 2), epigenetic_features)
            ], 
            remainder='passthrough')
        )
    ])

    X = pipeline.fit_transform(X)

    return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    X = normalize_features(args.input)
    
    X.to_csv(args.output, index=False, sep = '\t')