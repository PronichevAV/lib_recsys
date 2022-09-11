import pandas as pd
import uvicorn
from fastapi import FastAPI
from rectools.exceptions import NotFittedError

from config_loader import ConfigLoader
from data_pipeline import DataPipeline
from models.bm25_model import BM25Model
from models.popular_model import PopModel

app = FastAPI()
config = ConfigLoader()
popular_model = PopModel(config=config.common)
bm25_model = BM25Model(config=config.common, model_config=config.model)
data_pipeline = DataPipeline(data_config=config.data)
dataset = data_pipeline.dataset
user_ids = data_pipeline.user_ids


@app.get("/info")
def send_info():
    return {
        'message': 'Сервис предназначен для получения рекомендаций пользователей библиотеки. Для начала работы обучите модель.'
    }


@app.get("/fit")
def fit_model():
    bm25_model.fit(dataset)
    popular_model.fit(dataset)
    return {'message': 'Модели обучены'}


@app.get("/predict")
def predict():
    try:
        warm_recos = bm25_model.predict(user_ids=user_ids, dataset=dataset)
        cold_users = list(warm_recos.groupby(by='user_id').sum().query('rank < 210').index) + list(
            set(user_ids) - set(warm_recos['user_id']))
        cold_recos = popular_model.predict(user_ids=cold_users, dataset=dataset)
        full_recos = pd.concat([warm_recos[~warm_recos['user_id'].isin(cold_users)], cold_recos])
        submission = full_recos.loc[:, ['user_id', 'item_id']]
        submission.rename(columns={'user_id': 'chb', 'item_id': 'sys_numb'}, inplace=True)
        submission.to_csv(config.data.submissions_path + "/submission.csv", index=False, sep=';')
        return {'message': 'Предсказание получено', 'prediction_folder': config.data.submissions_path}
    except NotFittedError:
        return {'message': 'Предсказание не получено. Обучите модель (/fit)'}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
