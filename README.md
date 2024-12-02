# mini-autograd
A mini deep learning library with an automatic differentiation engine for tensors. This library implements backpropagation over a dynamically constructed DAG, along with a lightweight neural networks module.

### Setup (Recommended to use Python version : 3.9.20)
#### Step 1: It's recommended to use Python version 3.9.20. You can use [pyenv](https://github.com/pyenv/pyenv) to manage Python versions and virtual environments.

#### Step 2: Create a virtual environment at the project root.
```bash
pyenv virtualenv env-3.9.20
```

#### Step 3: Install all dependencies needed.
```bash
pip install -r requirements.txt
```

#### Step 4: Sanity Check (pytest)
```bash
pytest -s
```
