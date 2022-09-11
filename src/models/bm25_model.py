from implicit.nearest_neighbours import BM25Recommender
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel

from config_loader import BM25Config, CommonConfig
from models.abstract_model import AbstractModel


class BM25Model(AbstractModel):
    def __init__(self, config: CommonConfig, model_config: BM25Config):
        self.n_recos = config.n_recos
        self.model = ImplicitItemKNNWrapperModel(model=BM25Recommender(K=model_config.bm25_k,
                                                                       K1=model_config.bm25_k1,
                                                                       B=model_config.bm25_b,
                                                                       ))

    def fit(self, dataset: Dataset):
        self.model.fit(dataset)

    def predict(self, user_ids, dataset: Dataset):
        recos = self.model.recommend(
            users=user_ids,
            dataset=dataset,
            k=self.n_recos,
            filter_viewed=True
        )
        return recos
