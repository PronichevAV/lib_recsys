from pydantic import BaseModel


class ConfigLoader:
    @property
    def data(self):
        return DataConfig.parse_file('configs/data_config.json')

    @property
    def common(self):
        return CommonConfig.parse_file('configs/config.json')

    @property
    def model(self):
        return BM25Config.parse_file('configs/bm25_model_config.json')


class BM25Config(BaseModel):
    bm25_k: int = 100
    bm25_k1: float = 0.05
    bm25_b: float = 0.1


class CommonConfig(BaseModel):
    n_recos: int = 20


class DataConfig(BaseModel):
    items_path: str = "data/train/items.csv"
    users_path: str = "data/train/users.csv"
    interactions_path: str = "data/train/train_transactions_extended.csv"
    submissions_path: str = "data/submissions"
