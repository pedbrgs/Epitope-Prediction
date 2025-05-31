import copy
from abc import ABC, abstractmethod
from typing import Callable

from sklearn.base import ClassifierMixin


class BaselineFeatureSelector(ABC):
    """An abstract class for a baseline feature selection algorithm.

    Attributes
    ----------
    estimator : ClassifierMixin
        A fitted classifier with `predict` and `predict_proba` methods.
    eval_function : Callable
        A function with signature `eval_function(y_true, y_pred)` that returns a scalar score.
    random_state : int, optional
        Random seed used to initialize the estimator, by default 42.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        eval_function: Callable,
        random_state: int = 1234
    ):
        """Init baseline class."""
        self.base_estimator = estimator
        self.eval_function = eval_function
        self.random_state = random_state

    def _reset_estimator(self) -> ClassifierMixin:
        """Reset a fitted estimator to an unfitted estimator."""
        return copy.deepcopy(self.base_estimator)

    @abstractmethod
    def tune(self):
        """Tune the hyperparameters of the feature selection method."""
        pass

    @abstractmethod
    def fit(self):
        """Fit the feature selection method with the best hyperparameters."""
        pass
