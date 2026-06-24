from typing import Dict, List

from tools.data import Raw
import xgboost
import torch
import sklearn


class Pipeline:
    def __init__(
        self, rolling_dates: List[List[str]], test_dates: List[str], model: any
    ):
        self.rolling_dates = rolling_dates
        self.test_dates = test_dates
        self.model = model

    def train(self, params: Dict = {}, data_loader=Raw.load_date):
        pass
