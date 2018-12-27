# DRAGONFIST
### Dimensionally Reduced Adversarial Generations on Networks Filtered by Integrated Static Transformations

This project examines the effects of applying transformations to images in the context
of image classification neural networks through the use of working protoypes. We investigate how
using filtered images during training and inference affects
the classification accuracy and resilience to adversarial
attacks. The rationale supporting this methodology is that
transforming the input images should obstruct or diffuse
the intended perturbations of an adversarial example,
while still preserving the significant features of the image
so the model can effectively generalize and classify. Our
findings indicate a marginal increase in classification
accuracy and a significant improvement in adversarial
robustness for the ensemble model against both black-box and white-box attacks.

The models are implemented using Tensorflow + Keras, and the attack experiments utilize the [Cleverhans](https://github.com/tensorflow/cleverhans) library.

