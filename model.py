import typing
import pickle
import pandas as pd
import numpy as np
import logging
from tqdm import trange

from .settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, CATEGORICAL_STE_FEATURES, \
    TARGET

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
# from raif_hack.data_transformers import SmoothedTargetEncoding

import torch
from torch import save, load, tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from torch.nn import Sequential

logger = logging.getLogger(__name__)


class BenchmarkModel(Module):
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(self, numerical_features: typing.List[str],
                 ohe_categorical_features: typing.List[str],
                 ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
                 model_params: typing.Dict[str, typing.Union[str, int, float]]):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.num_features),
            ('ohe', OneHotEncoder(), self.ohe_cat_features),
            ('ste', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
             self.ste_cat_features)])

        super(BenchmarkModel, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(70, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)

        layers = [
            self.hidden1, self.act1, self.hidden2, self.act2, self.hidden3
        ]
        self.model = Sequential(*layers)
        self.model.double()
        self._is_fitted = False
        self.corr_coef = 0

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        predictions = self.model(torch.tensor(X_manual))
        deviation = ((y_manual - predictions) / predictions).median()
        self.corr_coef = deviation

    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X

    def fit(self, X_offer: pd.DataFrame, y_offer: pd.Series,
            X_manual: pd.DataFrame, y_manual: pd.Series):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости

        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        logger.info('Fit nn')
        # self.pipeline.fit(X_offer, y_offer, model__feature_name=[f'{i}' for i in range(70)],
        #                   model__categorical_feature=['67', '68', '69'])
        # xzd = self.preprocessor.fit_transform(X_offer, y_offer)
        # print(xzd)
        # print(xzd.shape)
        X_offer = self.preprocessor.fit_transform(X_offer)
        X_manual = self.preprocessor.fit_transform(X_manual)
        criterion = MSELoss()
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        for epoch in trange(1):
            # enumerate mini batches
            for inputs, targets in zip(X_offer, y_offer):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                # print(tensor(inputs).double())
                yhat = self.model(torch.tensor(inputs))
                # calculate loss
                # print(yhat)
                # print(yhat.shape)
                # print(targets)
                loss = criterion(yhat, torch.tensor(targets).double())
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

        logger.info('Find corr coefficient')
        # self._find_corr_coefficient(X_manual, y_manual)
        # logger.info(f'Corr coef: {self.corr_coef:.2f}')
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            X = self.preprocessor.fit_transform(X)
            # print(torch.from_numpy(X))
            predictions = self.model(torch.from_numpy(X))
            corrected_price = predictions * (1 + self.corr_coef)
            return corrected_price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        """
        save(self.model.state_dict(), path)
        # with open(path, "wb") as f:
        #     pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        # with open(path, "rb") as f:
        #     model = pickle.load(f)
        model = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                              ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
