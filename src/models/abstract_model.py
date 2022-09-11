from abc import abstractmethod, ABC


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        """Метод, реализующий обучение модели"""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Метод, реализующий предсказание модели"""
        pass
