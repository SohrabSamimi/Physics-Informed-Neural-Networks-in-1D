# Physics-Informed Neural Networks in 1D

## Run the notebook

The setup below was verified on Linux with Python 3.12. It uses CPU-only
PyTorch and requires no GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

In JupyterLab, open `Physics-Informed Neural Networks.ipynb` with the Python 3
kernel and run all cells. To execute it from the terminal and write the results
to a temporary directory, run this instead of starting JupyterLab:

```bash
output_dir=$(mktemp -d)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
  --output-dir="$output_dir" 'Physics-Informed Neural Networks.ipynb'
```

PINNs were developed by Raissi et al. in 2017 in this article: https://arxiv.org/abs/1711.10561.

Standard Neural Networks (NNs) generally fail at solving even simple linear PDEs. PINNs use NNs and incorporate the PDE inside the

loss function, as we will see a few steps later.

In some sense, PINNs are "semi-supervised" meaning that the equation drives the learning process (PINNs are supervised "by the equation").

This repository first shows how NNs fail to learn a simple linear ODE outside the training set.

We train a baseline Neural Network to learn the cosine function.

We see here under a very good performance on the train set but not on the test set:

![Baseline_NN_cosine](https://github.com/SohrabSamimi/Physics-Informed-Neural-Networks-example/assets/58103877/8e6d8f5b-0f7d-417c-9f96-f455f8145983)

Afterwards, we take a completely different approach and we see the cosine function $x \longmapsto cos(x)$ as the solution of a second order differential equation, that is the 

solution to:

$y''=y$, 

with initial conditions:

$y(0)=1$, 

$y'(0)=0$

We use Physics-Informed Neural Networks ($PINNs$) to learn it using only 15 "equation" points. "Equation points" here

indicate we don't need data at those points (knowing the ODE verified by data is enough).

The central idea of $PINNs$ is to replace $y$ by a neural network $y_\theta$ and minimize the following loss:

$L = MSE(y_\theta''-y_\theta) + (y_\theta(0)-1)² + y_\theta'(0)²$.

If $L$ is close enough to $0$, it means that all the terms in the sum above are close to $0$, thus the differential

equation is (almost) satisified.

To do this, we do not need ANY data about the solution $y$ except for $y(0)$ and $y'(0)$, which is a major advantage compared 

to classical NNs, which require a lot of data.

We see that the $PINNs$ prediction and the true Cosine agree very well even outside the training set (15 points):

after training the model, we have an average error of $5.10^{-6}$ on a test set of 500 points.

![PINNS_COSINE](https://github.com/SohrabSamimi/Physics-Informed-Neural-Networks-example/assets/58103877/3e3350aa-18dc-4e3a-9015-8e22306984fb)


The PINNS technique not only allows to get a good estimation of the function itself but also of its derivatives

as we see below.

We recall that we should get  $NN'(x) = -sin(x)$ and $NN''(x) = -cos(x)$

![1st 2nd derivatives](https://github.com/SohrabSamimi/Physics-Informed-Neural-Networks-example/assets/58103877/8dd8f52c-50d1-4059-8c10-f79c621841fa)




