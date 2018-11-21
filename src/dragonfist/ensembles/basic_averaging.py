from keras import Model
import numpy as np


class BasicAveragingEnsemble(Model):

    def __init__(self, *claws):
        super()._base_init("BasicAveragingEnsemble")
        self._claws = claws

    def evaluate(self, x=None, y=None, **kwargs):
        num_samples = x.shape[0]

        predictions = self.predict(x)

        num_correct = 0
        for i in range(num_samples):
            if predictions[i].argmax() == y[i].argmax():
                num_correct += 1

        return num_correct / num_samples

    def predict(self, x, **kwargs):
        predictions = np.array([claw.predict(x) for claw in self._claws])
        avg = np.zeros_like(predictions[0])
        for p in predictions:
            avg += p
        return avg / len(self._claws)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
                validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
                steps_per_epoch=None, validation_steps=None, **kwargs):
        """Do nothing - no training required in this model"""

