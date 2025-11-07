---

#  Medical Insurance Cost Prediction using Neural Networks (PyTorch)

##  Overview

This project builds a **multi-layer feedforward neural network** using **PyTorch** to predict **individual medical insurance costs** based on demographic and health-related factors.
The model is trained and evaluated on the **Insurance Dataset**, a widely used dataset for regression-based cost prediction tasks.

---

##  Objective

To predict the **medical insurance charges (`charges`)** for an individual using their personal and lifestyle attributes.

| Feature    | Description                                                                       |
| ---------- | --------------------------------------------------------------------------------- |
| `age`      | Age of the primary beneficiary                                                    |
| `sex`      | Gender of the insurance holder                                                    |
| `bmi`      | Body Mass Index (BMI = weight / height²)                                          |
| `children` | Number of dependents covered                                                      |
| `smoker`   | Smoking status (`yes` / `no`)                                                     |
| `region`   | Residential area in the U.S. (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges`  | Target variable — individual medical cost billed                                  |

---

##  Methodology

### 1. Data Preprocessing

* The dataset is loaded using **Pandas**.
* The target variable `charges` is separated from the input features.
* Categorical columns (`sex`, `smoker`, `region`) are **one-hot encoded** using `pd.get_dummies()` with `drop_first=True`.
* The data is split into **training (80%)** and **testing (20%)** sets using `train_test_split()`.
* Numerical values are **standardized** using `StandardScaler()` to ensure stable training.

### 2. Model Architecture

The neural network (`InsuranceNN`) is defined using **PyTorch’s `nn.Module`**, with the following layers:

| Layer          | Description                           |
| -------------- | ------------------------------------- |
| Input          | Size = number of input features       |
| Hidden Layer 1 | 128 neurons + ReLU activation         |
| Hidden Layer 2 | 64 neurons + ReLU activation          |
| Hidden Layer 3 | 32 neurons + ReLU activation          |
| Output Layer   | 1 neuron (regression output for cost) |

**Loss Function:** Mean Squared Error (MSE)
**Optimizer:** Adam (learning rate = 0.001)
**Epochs:** 300

---

##  Training Process

* During each epoch:

  1. The model performs a **forward pass** to compute predictions.
  2. The **loss** between predictions and actual values is calculated.
  3. Gradients are **backpropagated**, and weights are updated using the Adam optimizer.
* Validation loss is monitored every 50 epochs.

Example training output:

```
Epoch [50/300] - Train Loss: 23752.1133 | Val Loss: 31502.2500
Epoch [100/300] - Train Loss: 15492.7324 | Val Loss: 21034.9238
...
```

---

##  Evaluation

After training, the model’s performance is evaluated using **Mean Absolute Error (MAE)**:

[
\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
]

Example output:

```
Mean Absolute Error (MAE): 2521.68
```

