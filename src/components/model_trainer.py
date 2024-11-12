import os
import sys
from dataclasses import dataclass
#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, model_evaluate

@dataclass
class model_trainer_config:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and testing data input")

            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                #"CatBoosting Classfier": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict=model_evaluate(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info("Best model is determined by training and testing ")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(xtest)

            r2score = r2_score(ytest, predicted)

            return r2score
        
        except Exception as e:
            raise CustomException(e, sys)

