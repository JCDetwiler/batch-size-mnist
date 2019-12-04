# batch-size-mnist
Reproducing results from Smith et al.'s 2018 batch size paper on a smaller CNN

This is a [CS 5824: Advanced Machine Learning][2] group project for reproducing
the results of a research paper. We intend to test Smith et al.'s results
from [Don't Decay the Learning Rate, Increase the Batch Size][1], by seeing
how well the technique transfers to a smaller CNN on the MNIST dataset as
opposed to ImageNet.


Figures

![vanilla sgd results](/vanilla_sgd.png)

![adam results](/adam.png)

![sgd with momentum results](/sgd_with_momentum.png)


[1]: https://arxiv.org/abs/1711.00489
[2]: http://courses.cs.vt.edu/cs5824/Fall19/project.html
