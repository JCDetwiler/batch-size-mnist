# Reproduction of "Don't Decay the Learning Rate, Increase the Batch Size"
Reproducing results from Smith et al.'s 2018 batch size paper on a smaller CNN.

This is a [CS 5824: Advanced Machine Learning][9] group project for reproducing
the results of a research paper. We intend to test Smith et al.'s results
from [Don't Decay the Learning Rate, Increase the Batch Size][2], by seeing
how well the technique transfers to a smaller CNN on the MNIST dataset as
opposed to ImageNet.

---

# Report

## Original paper

Smith et al. report both theoretical and experimental evidence of a method to 
reduce training time of a neural network without compromising its accuracy: 
instead of decaying the learning rate, increase the batch size. The authors 
train Wide ResNet, Inception-ResNet-V2, and ResNet-50 models, reporting that 
their procedure is successful with SGD, SGD with momentum, Nesterov momentum, 
and Adam optimizers.

The authors describe a "noise scale" in their analysis to support the claim. The 
noise scale "controls the magnitude of the random fluctuations in the training 
dynamics". They argue that decaying the learning rate is one way of reducing 
noise, and show that increasing batch size is an equivalent way of accomplishing 
this same objective. In their experiments, the authors test different training 
schedules and measure the validation accuracy and number of parameters updates 
needed to their models.

## Methodology

Intel Core i5-6600K CPU (3.5GHZ) with 16GB RAM is the hardware used to run our 
experiments. We used an Anaconda environment with Python 3.7.5 and ran 
experiments in Jupyter Notebooks. Our backend was Keras 2.2.4 and TensorFlow 
1.15.0. All dependencies can be installed with

    conda create -n EnvName python=3.7.5
    conda install -c conda-forge keras=2.2.4 tensorflow=1.15.0 numpy mnist jupyter

We selected [Victor Zhou's CNN][1] that trains on MNIST as a baseline model that 
is known to classify handwritten digits well, and made modifications to this 
model to compare with the authors' experiments.

The authors note that ghost batch normalization is necessary for their 
experiments, so we set up a `BatchNormalization` layer with virtual batch size 
10 and batch size 100 to precede the first (and only) CNN layer of our network. 
TensorFlow supports this only through the `tf.keras` libraries and not under 
just `keras`.

Second, since the authors test 3 different training schedules in their 5.1 
results, we establish callbacks that, over 6 epochs, employ their schedules in 
steps of 2 epochs. Since the model can be trained quickly, this is sufficient 
to observe the results being measured. TensorFlow and Keras have yet to accept 
pull requests that enable batch size callbacks, so we iteratively call `fit` to 
our model with different batch sizes to emulate this training schedule.

To reproduce the experiments, we tried the same optimizers used in their 5.1 
methodology and apply each of the 3 training schedules. We select our learning 
rates for SGD with respect to the training schedule (e.g. we expect that 
"decaying learning rate" should take a higher initial learning rate, and so on). 
For SGD, we tried Vanilla SGD and SGD with momentum as was done in the paper. 
Our parameters for Adam are the default, similar to the authors'.

## Results

The following is the baseline result for training a basic convolutional neural network with Adam on Mnist. The baseline took 49.19 seconds to finish. The final validation accuracy was 0.9744.

![baseline adam results](/Plots/baseline_adam.png)

We could reproduce the same results that the authors got in their experiments. 
The following is a comparison of the authors' results with ours.

| vanilla sgd results  | paper's vanilla sgd results |
| ------------- | ------------- |
| ![vanilla sgd results](/Plots/vanilla_sgd.png)  | ![paper's vanilla sgd results](/Plots/paper_vanilla_sgd.PNG)  |

| adam results  | paper's adam results |
| ------------- | ------------- |
| ![adam results](/Plots/adam.png)  | ![paper's adam results](/Plots/paper_adam.PNG)  |

| sgd with momentum results  | paper's sgd with momentum results |
| ------------- | ------------- |
| ![sgd with momentum results](/Plots/sgd_with_momentum.png)  | ![paper's sgd with momentum results](/Plots/paper_sgd_momentum.PNG)  |

Increasing the batch size instead of decaying the learning rate indeed have 
similar validation accuracy as was reported in the paper, within 1% 
difference for all setups.

Next, these are our results for measuring the training time.

| vanilla sgd training times for each schedule  |
| ------------- |
| ![vanilla sgd training time](/Plots/vanillaSGD_time.png)  |

| adam training times for each schedule  |
| ------------- |
| ![adam training time](/Plots/adam_time.png)  |

| sgd with momentum training times for each schedule  |
| ------------- |
| ![sgd with momentum training time](/Plots/SGDmomentum_time.png)  |

We found all three setups, Vanilla SGD, Adam, and SGD with momentum, follow the 
same pattern. Decaying the learning rate had the longest training time, followed 
by the hybrid schedule, and lastly increasing the batch size. The difference 
between the longest and the shortest training time was about 2-3 seconds.

We noticed that the training time for each epoch when increasing the batch size 
at each step noticeably decreased the training time as can be seen below.

![training_time](/Plots/training_screenshot.png)

## Relevant Background Papers

In our paper, Smith et al. reference Goyal et al. for motivation of their 
research as well as for implementing a similar SGD set up that Goyal et al. used 
[[5]]. Goyal et al. introduce an optimization method of increasing the mini batch 
size for momentum SGD using parallel computing for each mini batch. Before their 
work, small mini batch sizes were used, which have long training times. Goyal et 
al. introduce a scaling rule that they adapt in their experiments to increase 
the mini batch size without sacrificing generalization accuracy. According to 
the scaling rule, when increasing the batch size, a larger learning rate should 
accordingly be used. Goyal et al. also introduce the employment of a warm-up 
stage during pre-training for better results. As a result, they could 
significantly reduce the training time for ImageNet using a mini batch size of 
8192 and parallelization. Smith et al. replicates the set up by Goyal et al. in 
their attempt to train ImageNet within 2500 parameter updates. Goyal et al.'s 
research provided a good basis for claiming that the use of large batch sizes 
reduces training time.


The authors of our paper reference Hoffer et al.'s algorithm "Ghost Batch 
Normalization" which was introduced in [Train longer, generalize better][3]. 
Hoffer et al. analyze the generalization gap that happens with training on large 
batch sizes. They empirically show that the generalization gap can be eliminated 
by adapting the training regime used, which our authors take advantage of. Their 
ghost batch normalization procedure decreases the gap without increasing 
parameters updates to the model during training. Part of their work shows that 
SGD can converge to strongly convex minima when altering the batch size as part 
of the training schedule.

We also looked at a paper previously published by our authors, [A Bayesian 
Perspective on Generalization and Stochastic Gradient Descent][4]. In the 
authors' theoretical argument in support of an increasing batch size training 
schedule, they reference this previous paper which demonstrates an optimal batch 
size exists. They further argue that by solving SGD as a differential equation, 
it has direct correspondence to the "noise scale" mentioned in this paper. This 
connects to ghost batch normalization by their claim that noise introduced by 
mini-batches helps SGD find minima that will generalize well. They offer 
empirical evidence to corroborate the proportionality between batch size and 
noise scale.

## Analysis of Results

We see comparable accuracies across the different training schedules while 
increasing the batch size reduces training time. This training time comparison 
depends on consistency with the hardware used; because we didn't use GPUs and it 
was multiprocessed all on the same machine, the comparison across experiments is 
fair. While the accuracies differ up to 1%, we expect that hyperparameter 
adjustment or even rerunning the experiments could close the gap. Our results 
appear to corroborate the authors' claims: we were able to trade off decaying 
the learning rate in favor of increasing the batch size and still saw good 
accuracy while decreasing training time.

We'd like to note that the experiments lack a clear independent variable. The 
training schedules are the independent variable being tested; however, there are 
confounding variables in this---namely, the learning rate and the batch size. By 
adjusting both variables in each training schedule, it's hard to evaluate the 
fairness of the experiments performed. Consider that the learning rate and batch 
size are not independent on their effects toward the model's ability to 
generalize on the dataset.

Overall, we were able to reproduce some of the results from the authors on a 
different CNN. We suspect it would be possible to reproduce more if the entire 
set of experiments were implemented.

## References

1. [Don't Decay the Learning Rate, Increase the Batch Size][2]
2. [Train longer, generalize better: closing the generalization gap in large batch training of neural networks][3]
3. [A Bayesian Perspective on Generalization and Stochastic Gradient Descent][4]
4. [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour][5]
5. [Keras for Beginners: Implementing a Convolutional Neural Network][1]
6. [Using Learning Rate Schedules for Deep Learning Models in Python with Keras][6]
7. [How to re-initialize Keras model weights][7]
8. [How to measure training time between epochs][8]


[1]: https://victorzhou.com/blog/keras-cnn-tutorial/
[2]: http://arxiv.org/abs/1711.00489
[3]: http://arxiv.org/abs/1705.08741
[4]: http://arxiv.org/abs/1710.06451
[5]: http://arxiv.org/abs/1706.02677
[6]: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
[7]: https://www.codementor.io/nitinsurya/how-to-re-initialize-keras-model-weights-et41zre2g
[8]: https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
[9]: http://courses.cs.vt.edu/cs5824/Fall19/project.html
