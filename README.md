# Linear Regression - Energy Consumption Prediction

A from-scratch implementation of Simple and Multiple Linear Regression using gradient descent to predict building energy consumption.

---

## Dataset

The dataset contains 5 columns:

| Feature | Description |
|---|---|
| Square Footage | Size of the building |
| Number of Occupants | How many people occupy the building |
| Appliances Used | Number of appliances in use |
| Average Temperature | Average temperature in the building |
| Energy Consumption | Target variable (kWh) |

---

## Models Trained

6 models in total:

- Simple Linear Regression — Square Footage
- Simple Linear Regression — Number of Occupants
- Simple Linear Regression — Appliances Used
- Simple Linear Regression — Average Temperature
- Multiple Linear Regression — All Features
- Multiple Linear Regression — All Features except Average Temperature

---

## Results Summary

| Model | MSE | RMSE |
|---|---|---|
| Square Footage | 347,708.84 | 589.67 |
| Number of Occupants | 760,853.12 | 872.27 |
| Appliances Used | 785,062.79 | 886.04 |
| Average Temperature | 869,167.22 | 932.29 |
| **All Features** | **165,639.57** | **406.99** |
| All Features (no Temp) | 167,048.80 | 408.72 |

**Best single feature:** Square Footage  
**Best overall model:** Multiple Linear Regression with all features

---

## Plots

### Learning Rate vs MSE        \t                                        Epochs vs MSE

<!-- Add your LR vs MSE plot here -->
<img width="1400" height="500" alt="LR, epochs vs MSE" src="https://github.com/user-attachments/assets/85a245af-5cb8-4333-9778-e7df59922dee" />

---

## Implementation Details

- Normalization: Min-Max Scaling to [0, 1]
- Weight initialization: m = 0, c = 0
- Learning rates tried: 0.001, 0.01, 0.04, 0.06, 0.08, 0.1, 0.11, 0.15, 0.17, 0.2
- Epochs tried: 30, 50, 60, 100, 200, 300
- Best learning rate: 0.1
- Best epochs: 1000

---

## How to Run

```bash
pip install numpy pandas matplotlib
python main.py
```

---

## Files

- `main.py` — main code
- `report.pdf` — full report with plots and conclusions
- `energycons.csv` — dataset

---

## Author

**Aly Mahmoud Aly Hassan**  
