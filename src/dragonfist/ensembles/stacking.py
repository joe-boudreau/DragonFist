import numpy as np
import math
from sklearn.linear_model import LogisticRegression


class Stacking:

    def __init__(self, *claws):
        self._claws = claws

        self._ensemble = LogisticRegression(random_state=0, solver="lbfgs", multi_class="multinomial", max_iter=200)

    @property
    def num_claws(self):
        return len(self._claws)

    @property
    def claws(self):
        return self._claws

    def evaluate(self, x, y):
        claws_test_preds = self._claw_predictions(x)

        y = np.array([yi.argmax() for yi in y[:, ]])
        score = self._ensemble.score(claws_test_preds, y)

        return score

    def predict(self, x):
        return self._ensemble.predict_proba(self._claw_predictions(x))

    def fit(self, x, y, validation_split=0.2):

        self._num_classes=y.shape[1]

        # Validation Split
        split_i = math.floor(x.shape[0]*(1-validation_split))
        x = x[:split_i]
        y = y[:split_i]
        x_test = x[-split_i:]
        y_test = y[-split_i:]

        claws_train_preds = self._claw_predictions(x)

        # Convert categorical matrix back into 1D output array (Logistic Regression does not support one-hot)
        y = np.array([yi.argmax() for yi in y[:, ]])

        self._ensemble.fit(claws_train_preds, y)

    def _claw_predictions(self, x):
        # Generate train and test sets of shape (number of samples, number of categories, number of claws)
        claws_preds = np.empty((x.shape[0], self._num_classes, len(self._claws)), float)
        for i in range(len(self._claws)):
            claws_preds[:, :, i] = self._claws[i].predict(x)

        # Reshape inputs for ensemble model as matrix of (number of categories * number of claws)
        return claws_preds.reshape(claws_preds.shape[0], claws_preds.shape[1]*claws_preds.shape[2])
