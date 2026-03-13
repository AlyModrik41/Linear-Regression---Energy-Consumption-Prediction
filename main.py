import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('assignment1dataset.csv')

LEARNING_RATES = [0.001, 0.01, 0.04, 0.06, 0.08, 0.1,0.11,0.15, 0.17, 0.2]
EPOCHS_LIST = [30, 50, 60, 100, 200,300]
BEST_LR = 0.1
BEST_EPOCHS = 1000
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Epochs vs MSE & Learning Rate vs MSE', fontsize=13, fontweight='bold')

y = df['Energy Consumption']
features = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']

def simple_linear_regression(X, y, feature_name, axes):
    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()
    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)
    n = float(len(X))
    print(f'=========================== Epochs in {feature_name} ================================')

    # Plot 1: Epochs vs MSE — fixed best_lr
    mse_per_epoch = []
    for epochs in EPOCHS_LIST:
        m, c = 0, 0
        for _ in range(epochs):
            y_pred = m * X_norm + c
            D_m = (-2/n) * np.sum((y_norm - y_pred) * X_norm)
            D_c = (-2/n) * np.sum(y_norm - y_pred)
            m = m - BEST_LR * D_m
            c = c - BEST_LR * D_c
        prediction_norm = m * X_norm + c
        predictions = prediction_norm * (y_max - y_min) + y_min
        mse = (1/n) * np.sum((y - predictions) ** 2)
        rmse=np.sqrt(mse)
        mse_per_epoch.append(mse)
        print(f"MSE for {feature_name}, Epochs: {epochs} -> {mse:.2f}, RMSE -> {rmse:.2f}")
    axes[0].plot(EPOCHS_LIST, mse_per_epoch, marker='*', label=feature_name)
    print(f'=========================== LRs in {feature_name} ================================')

    # Plot 2: LR vs MSE — fixed best_epochs
    mse_per_lr = []
    for lr in LEARNING_RATES:
        m, c = 0, 0
        for _ in range(BEST_EPOCHS):
            y_pred = m * X_norm + c
            D_m = (-2/n) * np.sum((y_norm - y_pred) * X_norm)
            D_c = (-2/n) * np.sum(y_norm - y_pred)
            m = m - lr * D_m
            c = c - lr * D_c
        prediction_norm = m * X_norm + c
        predictions = prediction_norm * (y_max - y_min) + y_min
        mse = (1/n) * np.sum((y - predictions) ** 2)
        mse_per_lr.append(mse)
        rmse=np.sqrt(mse)
        print(f"MSE for {feature_name}, Learning Rate: {lr} -> {mse:.2f}, RMSE -> {rmse:.2f} ")
    axes[1].plot(LEARNING_RATES, mse_per_lr, marker='o', label=feature_name)


def multiple_linear_regression(X, y, axes,feature_names):

    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()
    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)
    n = float(len(y))
    print(f'=========================== Epochs in {feature_names} ================================')
    # Plot 1: Epochs vs MSE — fixed best_lr
    mse_per_epoch = []
    for epochs in EPOCHS_LIST:
        m = np.zeros(X.shape[1])
        c = 0
        for _ in range(epochs):
            y_pred = X_norm.dot(m) + c
            error = y_norm - y_pred
            D_m = (-2/n) * X_norm.T.dot(error)
            D_c = (-2/n) * np.sum(error)
            m = m - BEST_LR * D_m
            c = c - BEST_LR * D_c
        prediction_norm = X_norm.dot(m) + c
        predictions = prediction_norm * (y_max - y_min) + y_min
        mse = (1/n) * np.sum((y.values - predictions) ** 2)
        mse_per_epoch.append(mse)
        rmse=np.sqrt(mse)
        print(f"MSE for {feature_names}, Epochs: {epochs} -> {mse:.2f}, RMSE -> {rmse:.2f} ")
    axes[0].plot(EPOCHS_LIST, mse_per_epoch, marker='s', linestyle='dashed', linewidth=2, label=feature_names)
    print(f'=========================== LRs in {feature_names} ================================')
    # Plot 2: LR vs MSE — fixed best_epochs
    mse_per_lr = []
    for lr in LEARNING_RATES:
        m = np.zeros(X.shape[1])
        c = 0
        for _ in range(BEST_EPOCHS):
            y_pred = X_norm.values.dot(m) + c
            error = y_norm.values - y_pred
            D_m = (-2/n) * X_norm.values.T.dot(error)
            D_c = (-2/n) * np.sum(error)
            m = m - lr * D_m
            c = c - lr * D_c
        prediction_norm = X_norm.values.dot(m) + c
        predictions = prediction_norm * (y_max - y_min) + y_min
        mse = (1/n) * np.sum((y.values - predictions) ** 2)
        mse_per_lr.append(mse)
        rmse=np.sqrt(mse)
        print(f"MSE for {feature_names}, Learning Rate: {lr} -> {mse:.2f}, RMSE -> {rmse:.2f} ")
    axes[1].plot(LEARNING_RATES, mse_per_lr, marker='s', linestyle='dashdot', linewidth=2, label=feature_names)

for feature in features:
    simple_linear_regression(df[feature], y, feature, axes)

X_multi = df[features]
multiple_linear_regression(X_multi, y, axes,'All Features')
X_multi_wotemp=X_multi.drop(columns=['Average Temperature'])
multiple_linear_regression(X_multi_wotemp,y,axes,'All Features, but Temperature')

print(f'Best Single Feature to Predict the Energy Consumption is: Square Footage according to its Mse which is equal to: 347708.84 ')

axes[0].set_title(f'Epochs vs MSE (lr={BEST_LR})')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('MSE')
axes[0].legend()

axes[1].set_title(f'Learning Rate vs MSE (epochs={BEST_EPOCHS})')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('MSE')
axes[1].legend()

plt.tight_layout()
plt.show()
