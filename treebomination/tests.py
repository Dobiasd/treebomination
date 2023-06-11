import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

from ._conversion import treebominate, tree_input_to_nn_input


class TestArgsCalls(unittest.TestCase):

    def test_foo_function_positional(self) -> None:
        # https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
        df_train = pd.read_csv("kaggle_house-prices-advanced-regression-techniques_train.csv", header=0)

        # Only the numerical columns
        columns = [
            "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
            "MasVnrArea",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
            "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
            "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
            "ScreenPorch",
            "PoolArea", "MiscVal", "MoSold", "YrSold"
        ]

        df_train = df_train[columns + ["SalePrice"]].dropna()
        X = df_train[columns].to_numpy()
        Y = df_train[["SalePrice"]].to_numpy().ravel()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=23)

        tr = DecisionTreeRegressor(max_depth=3)
        tr.fit(X_train, Y_train)

        print(export_text(tr))

        nn = treebominate(tr.tree_, len(columns))

        nn.compile(loss='mse', optimizer='nadam')
        tf.keras.utils.plot_model(
            nn, to_file='model.png',
            show_shapes=False, show_dtype=False, show_layer_names=False,
            rankdir='LR', expand_nested=True, dpi=96,
            layer_range=None, show_layer_activations=False, show_trainable=False
        )

        def eval_tr():
            tr_test_pred = tr.predict(X_test)
            print("tree score:", r2_score(Y_test, tr_test_pred))

        def eval_nn():
            nn_test_pred = np.squeeze(nn(tree_input_to_nn_input(X_test)).numpy())
            print("nn score:", r2_score(Y_test, nn_test_pred))

        eval_tr()
        eval_nn()
        nn.fit(tree_input_to_nn_input(X_train), Y_train, epochs=10)
        eval_nn()
