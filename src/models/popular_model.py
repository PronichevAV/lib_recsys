from rectools.dataset import Dataset
from rectools.models import PopularModel

from config_loader import CommonConfig
from models.abstract_model import AbstractModel


class PopModel(AbstractModel):
    def __init__(self, config: CommonConfig):
        self.n_recos = config.n_recos
        self.model = PopularModel()

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
