import numpy as np
from sklearn.preprocessing import MinMaxScaler
from . testing_functions import testing_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

"""
utility functions to make one-dimensional datasets
"""

x_dim = 1
vmax = 1
test_color = "#377eb8"
train_color = "Orange"
size = 10


def prepare_dataset(n_all_record=10, mode="sin", plot=True,
                    extra_high_ratio=0.1,
                    extra_low_ratio=0.1,
                    inner_test_ratio=0.2):
    low_threshold_id = int(n_all_record*extra_low_ratio)
    high_threshold_id = int(n_all_record*(1-extra_high_ratio))

    X_array = np.random.random((n_all_record, x_dim))
    X_array = np.sort(X_array, axis=0)

    scaler_X = MinMaxScaler(feature_range=(-vmax, vmax))
    #scaler_y = MinMaxScaler(feature_range=(-1,1))
    X = scaler_X.fit_transform(X_array.reshape(-1, x_dim))
    # y=scaler_y.fit_transform(y_array.reshape(-1,1))
    y_array = testing_function(X, mode=mode)
    y = y_array.reshape(-1, 1)

    te_low_X = X[:low_threshold_id]
    te_low_y = y[:low_threshold_id].reshape(-1)
    te_high_X = X[high_threshold_id:]
    te_high_y = y[high_threshold_id:].reshape(-1)

    inner_X = X[low_threshold_id:high_threshold_id]
    inner_y = y[low_threshold_id:high_threshold_id].reshape(-1)

    # print(low_threshold_id,high_threshold_id,extra_low_ratio,extra_high_ratio)

    train_size = 1-extra_high_ratio-extra_low_ratio-inner_test_ratio
    train_size = train_size/(inner_test_ratio+train_size)

    if train_size == 1:
        tr_X = inner_X
        tr_y = inner_y
        te_inner_X = np.array([])
        te_inner_y = np.array([])
    else:
        tr_X, te_inner_X, tr_y, te_inner_y = train_test_split(
            inner_X, inner_y, train_size=train_size)

    tr_y = tr_y.reshape(-1)
    te_inner_y = te_inner_y.reshape(-1)

    if plot:
        plt.figure(figsize=(2, 2), dpi=150)
        plt.scatter(te_low_X, te_low_y, label="Test", c=test_color, s=size)
        plt.scatter(te_high_X, te_high_y, c=test_color, s=size)
        plt.scatter(te_inner_X, te_inner_y, c=test_color, s=size)
        plt.scatter(tr_X, tr_y, label="Train", c=train_color, s=size)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   borderaxespad=0, fontsize=8)

    act_X = np.linspace(-1, 1, 100).reshape(-1, 1)
    act_y = testing_function(act_X, mode=mode)
    return tr_X, tr_y, te_inner_X, te_inner_y, te_low_X, te_low_y, te_high_X, te_high_y, act_X, act_y


def eval(ax, model, tr_X, tr_y, te_inner_X, te_inner_y, te_low_X, te_low_y, te_high_X, te_high_y, act_X, act_y, title="", plot=True):

    loss_list = []
    if plot:
        # plt.figure(figsize=(2,2),dpi=150)
        ax.set_title(title)
        ax.plot(act_X, act_y, label="Answer", c="gray", linewidth=1, alpha=0.5)
    for (X, y), label in zip([
        (te_inner_X, te_inner_y),
        (tr_X, tr_y),
        (te_low_X, te_low_y),
        (te_high_X, te_high_y)
    ],
            ("Test", "Train", "", "")):

        if X.shape[0] == 0:
            continue

        y_pred = model.predict(X)
        y_pred[np.where(y_pred != y_pred)] = 0

        if y_pred.shape[0] == 0:
            loss_list.append(None)
        else:
            loss_list.append(mean_squared_error(y, y_pred))

        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        # plt.figure()
        if plot:
            if label == "Train":
                c = train_color
            else:
                c = test_color

            ax.scatter(X, y_pred, label=label, c=c, s=size)
    # if plot:
        #ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',fontsize=8)

        #plt.legend([],[], frameon=False)
    return loss_list
