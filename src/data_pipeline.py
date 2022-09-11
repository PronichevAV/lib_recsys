import pandas as pd
from rectools.dataset import Dataset

from config_loader import DataConfig


class DataPipeline:
    def __init__(self, data_config: DataConfig):
        self.items_data = self.load_data(path=data_config.items_path,
                                         dtype={'author': str, 'bbk': str, 'izd': str, 'sys_numb': str, 'title': str,
                                                'year_izd': str})
        self.users_data = self.load_data(path=data_config.users_path,
                                         dtype={'age': str, 'chb': str, 'chit_type': str, 'gender': str})
        self.interactions_data = self.load_data(path=data_config.interactions_path,
                                                dtype={'chb': str, 'date_1': str, 'is_printed': str, 'is_real': str,
                                                       'source': str, 'sys_numb': str, 'type': str})
        self.dataset, self.user_ids = self.prepare_data()

    def prepare_data(self):
        interactions = self.interactions_data.copy()
        interactions['date_1'] = pd.to_datetime(interactions['date_1'], yearfirst=True)
        interactions.rename(columns={'chb': 'user_id', 'sys_numb': 'item_id', 'date_1': 'datetime'}, inplace=True)
        interactions['weight'] = 10

        users = self.users_data.copy()
        users.rename(columns={'chb': 'user_id'}, inplace=True)
        users['age'] = users['age'].apply(self.get_age_bins)
        users = users[['user_id', 'age', 'gender']].copy()
        users = users.loc[users["user_id"].isin(interactions["user_id"])].copy()
        users['age'] = users['age'].fillna('unknown_age')
        users['gender'] = users['gender'].apply(lambda x: 'unknown_gender' if x in ['не указан', 'отсутствует'] else x)

        user_features_frames = []
        for feature in ["gender", "age", "chit_type"]:
            feature_frame = users.reindex(columns=["user_id", feature])
            feature_frame.columns = ["id", "value"]
            feature_frame["feature"] = feature
            user_features_frames.append(feature_frame)
        user_features = pd.concat(user_features_frames)
        user_features = user_features.loc[~user_features['value'].isin(['не указан', 'отсутствует', 'нет данных']), :]
        user_features.reset_index(drop=True, inplace=True)

        dataset = Dataset.construct(interactions_df=interactions,
                                    user_features_df=user_features,
                                    cat_user_features=['gender', 'age']
                                    )
        return dataset, interactions['user_id'].unique()

    @staticmethod
    def load_data(path, dtype, sep=';', index_col=None):
        data = pd.read_csv(path, sep=sep, index_col=index_col, dtype=dtype)
        return data

    @staticmethod
    def get_age_bins(age):
        try:
            age = int(age)
            if 0 <= age <= 17:
                return 'age_0_17'
            elif 18 <= age <= 24:
                return 'age_18_24'
            elif 25 <= age <= 34:
                return 'age_25_34'
            elif 35 <= age <= 44:
                return 'age_35_44'
            elif 45 <= age <= 54:
                return 'age_45_64'
            elif age >= 65:
                return 'age_65_inf'
        except ValueError:
            return None
