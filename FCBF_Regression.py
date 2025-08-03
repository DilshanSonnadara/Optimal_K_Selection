import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

class FCBF_MI:
    """
    Fast Correlation-Based Filter (FCBF) using Mutual Information for regression tasks.

    Selects features that are:
      - Highly relevant to the target (MI >= delta)
      - Minimally redundant with each other (based on MI between features)

    Parameters
    ----------
    delta : float
        Minimum mutual information threshold with the target.

    discrete_features : bool or array-like of shape (n_features,), default=False
        Whether to treat features as discrete when computing MI.
    """

    def __init__(self, delta=0.0, discrete_features=False):
        self.delta = delta
        self.discrete_features = discrete_features
        self.selected_features_ = []

    def fit(self, X, y):
        """
        Fit the FCBF selector to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = pd.DataFrame(X)
        y = np.array(y)

        # Step 1: Compute MI between each feature and target
        mi_with_target = mutual_info_regression(
            X, y, random_state=0, discrete_features=self.discrete_features
        )

        # Filter features based on relevance threshold
        relevance = [(i, mi) for i, mi in enumerate(mi_with_target) if mi >= self.delta]
        relevance.sort(key=lambda x: x[1], reverse=True)

        selected = []

        # Step 2: Redundancy elimination
        while relevance:
            best_idx, best_mi = relevance.pop(0)
            selected.append(best_idx)

            new_relevance = []
            for idx, mi in relevance:
                # MI between selected best and current candidate feature
                # Use a single-element list [False] or [True] depending on feature type
                mi_between = mutual_info_regression(
                    X.iloc[:, [best_idx]], X.iloc[:, idx],
                    random_state=0,
                    discrete_features=[self.discrete_features[best_idx]]  # <== use only the correct flag
                )[0]


                # Keep if not too redundant
                if mi_between < mi:
                    new_relevance.append((idx, mi))

            relevance = new_relevance

        self.selected_features_ = selected
        return self

    def transform(self, X):
        """
        Return only selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_new : array of shape (n_samples, n_selected_features)
        """
        return X[:, self.selected_features_]

