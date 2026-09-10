# Linear Regression from Scratch

An educational implementation of simple linear regression and gradient descent using NumPy. The repository exposes prediction, mean-squared-error, and gradient calculations as small modules, then compares the learned line with scikit-learn's implementation.

## What is included

- A five-point synthetic regression dataset.
- A linear prediction function, `y = mx + b`.
- Mean squared error implemented from first principles.
- Analytical gradients for slope and intercept.
- Gradient-descent training with a convergence tolerance.
- A loss-versus-slope visualization.
- A side-by-side comparison with `sklearn.linear_model.LinearRegression`.

## Repository contents

| File | Purpose |
| --- | --- |
| `main.py` | Runs gradient-descent training and prints the learned parameters |
| `dataset.py` | Provides the sample `X` and `Y` values |
| `model.py` | Implements linear prediction |
| `loss.py` | Implements mean squared error |
| `gradient.py` | Computes gradients for slope and intercept |
| `plot.py` | Plots sample MSE values against slope |
| `compare.py` | Compares the custom predictions with scikit-learn |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
python main.py
python compare.py
python plot.py
```

The comparison script contains the parameters learned by the current training setup. Update them if the dataset or optimization settings change.
