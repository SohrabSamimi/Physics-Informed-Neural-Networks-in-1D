# Physics-Informed Neural Networks in 1D

This [notebook](Physics-Informed%20Neural%20Networks.ipynb) compares two ways
to learn the cosine on $[0, 2\pi]$. First, we train a regular neural network
on 16 cosine values and see how it behaves beyond the last training point.
Then we train a Physics-Informed Neural Network (PINN) of the same size using
the equation and its initial conditions.

## What the PINN learns

The cosine solves this ordinary differential equation:

$$
y''(x) + y(x) = 0, \qquad y(0) = 1, \qquad y'(0) = 0.
$$

The PINN uses 15 points spread across the full interval. At each point,
PyTorch computes the second derivative of the network output with respect to
its input. The loss combines the mean squared residual $y_\theta'' + y_\theta$
with a penalty for each initial condition. We use the cosine values only to
evaluate the PINN after training.

The notebook compares the two predictions on a grid of 500 points. It reports
mean squared errors on the full grid and beyond the last labeled training
point. That region tests extrapolation for the regular network, while the
PINN has equation points there. The final plots compare the PINN's derivatives
with $-\sin(x)$ and $-\cos(x)$.

This example is based on the PINN approach introduced by
[Raissi et al. (2017)](https://arxiv.org/abs/1711.10561).

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
