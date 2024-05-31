import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

def oja_k(X, m, eta, k):
    d, n = X.shape
    X = X.astype(np.float64)
    w_t = np.random.rand(d, k)
    w_t = w_t / np.linalg.norm(w_t)
    w_t = w_t.astype(np.float64)
    error_vec = []

    for s in tqdm(range(25), desc=f"Oja's algorithm using eta={eta}, k={k}"):
        count = 1
        w = w_t
        for t in range(1, m):
            i = np.random.randint(n)
            sample = np.expand_dims(X[:, i], axis=1)
            _w = w + (eta / count) * np.dot(sample, np.dot(sample.T, w))
            _w, r = np.linalg.qr(_w)
            w = _w

        if np.linalg.norm(X.T.dot(w_t)) > np.linalg.norm(X.T.dot(w)):
            error = 1 - np.linalg.norm(X.T.dot(w)) ** 2 / np.linalg.norm(X.T.dot(w_t)) ** 2

        else:
            error = 1 - np.linalg.norm(X.T.dot(w_t)) ** 2 / np.linalg.norm(X.T.dot(w)) ** 2

        error_vec.append(np.log10(error))
        w_t = w

    return error_vec


def vr_pca_k(X, m, eta, k):
    d, n = X.shape
    X = X.astype(np.float64)
    w_t = np.random.rand(d, k) - 0.5
    w_t = w_t / np.linalg.norm(w_t)
    w_t = w_t.astype(np.float64)

    error_vec = []
    for s in tqdm(range(25), desc=f"vr_pca algorithm"):

        u_t = np.matmul(X, np.matmul(X.T, w_t)) / n
        w = w_t

        for t in range(m):
            i = np.random.randint(n)
            sample = np.expand_dims(X[:, i], axis=1)
            res = sample.T.dot(w) - sample.T.dot(w_t)
            res2 = eta * (np.matmul(sample, res) + u_t)
            w_temp = w + res2
            w_temp = w_temp / np.linalg.norm(w_temp)
            w = w_temp

        if np.linalg.norm(X.T.dot(w_t)) > np.linalg.norm(X.T.dot(w)):
            error = np.log10(1 - np.linalg.norm(X.T.dot(w)) ** 2 / np.linalg.norm(X.T.dot(w_t)) ** 2)

        else:
            error = np.log10(1 - np.linalg.norm(X.T.dot(w_t)) ** 2 / np.linalg.norm(X.T.dot(w)) ** 2)
            w_t = w

        error_vec.append(error)
        if error < -10:
            return error_vec

    return error_vec


def power_iterations_k(X, eta, k):
    d, n = X.shape
    w_t = np.random.rand(d, k)
    w_t = w_t / np.linalg.norm(w_t)
    X = X.astype(np.float64)
    error_vec = []
    w_t = w_t.astype(np.float64)

    for s in tqdm(range(25), desc=f"Power iteration algorithm"):
        w = w_t

        A = 1 / n * X.dot(X.T)
        w = w + eta * A.dot(w)
        w = w / np.linalg.norm(w)

        if np.linalg.norm(X.T.dot(w_t)) > np.linalg.norm(X.T.dot(w)):
            error = 1 - np.linalg.norm(X.T.dot(w)) ** 2 / np.linalg.norm(X.T.dot(w_t)) ** 2

        else:
            error = 1 - np.linalg.norm(X.T.dot(w_t)) ** 2 / np.linalg.norm(X.T.dot(w)) ** 2
            w_t = w
        error_vec.append(np.log10(error))

    return error_vec


def vr_pca_better_k(X, m, eta, k):
    d, n = X.shape
    w_t = np.random.rand(d, k) - 0.5
    w_t = w_t / np.linalg.norm(w_t)
    X = X.astype(np.float64)
    w_t = w_t.astype(np.float64)
    error_vec = []

    for s in range(25):
        u_t = X.dot(X.T.dot(w_t)) / n
        w = w_t

        for t in range(m):
            i = np.random.randint(n)
            sample = np.expand_dims(X[:, i], axis=1)
            res = sample.T.dot(w) - sample.T.dot(w_t)
            res2 = (eta / (t + 1)) * (np.matmul(sample, res) + u_t)
            w_temp = w + res2
            w_temp, r = np.linalg.qr(w_temp)
            w = w_temp

        if np.linalg.norm(X.T.dot(w_t)) > np.linalg.norm(X.T.dot(w)):
            error = np.log10(1 - np.linalg.norm(X.T.dot(w)) ** 2 / np.linalg.norm(X.T.dot(w_t)) ** 2)

        else:
            error = np.log10(1 - np.linalg.norm(X.T.dot(w_t)) ** 2 / np.linalg.norm(X.T.dot(w)) ** 2)

        if error < -10:
            return error_vec
        w_t = w
        error_vec.append(error)

    return error_vec


def create_graphs_for_mnist_dataset_k(k):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    n = 60000
    d = 784
    X = np.reshape(X_train, (d, n))
    m = n
    r_h = (X ** 2).sum() / n
    eta = 1 / (r_h * np.sqrt(n))
    error1_oja_1 = oja_k(X, m, 1, k)
    error1_oja_3 = oja_k(X, m, 3, k)
    error1_oja_9 = oja_k(X, m, 9, k)
    error1_oja_27 = oja_k(X, m, 27, k)
    error1_oja_81 = oja_k(X, m, 81, k)
    error1_oja_243 = oja_k(X, m, 243, k)
    error_k_pca = vr_pca_k(X, m, eta, k)
    error_pi_k = power_iterations_k(X, eta, k)

    plt.figure(figsize=(7, 7))
    plt.plot(np.arange(1, len(error1_oja_1) + 1), error1_oja_1, label='oja eta=1')
    plt.plot(np.arange(1, len(error1_oja_3) + 1), error1_oja_3, label='oja eta=3')
    plt.plot(np.arange(1, len(error1_oja_9) + 1), error1_oja_9, label='oja eta=9')
    plt.plot(np.arange(1, len(error1_oja_27) + 1), error1_oja_27, label='oja eta=27')
    plt.plot(np.arange(1, len(error1_oja_81) + 1), error1_oja_81, label='oja eta=81')
    plt.plot(np.arange(1, len(error1_oja_243) + 1), error1_oja_243, label='oja eta=243')
    plt.plot(np.arange(1, len(error_pi_k) + 1), error_pi_k, label='power iteration')
    plt.plot(np.arange(1, len(error_k_pca) + 1), error_k_pca, label='vr-pca-k')
    plt.xlabel("# Data iterations")
    plt.ylabel('Loss')
    plt.ylim([-10, 0])
    plt.title(f"Convergence of different methods for k={k}")
    plt.legend()


def create_graphs_for_improved_algorithm():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    n = 60000
    d = 784
    X = np.reshape(X_train, (d, n))
    m = n
    r_h = (X ** 2).sum() / n
    eta = 1 / (r_h * np.sqrt(n))

    error1_oja_1 = oja_k(X, m, 1, 1)
    error1_oja_3 = oja_k(X, m, 3, 1)
    error1_oja_9 = oja_k(X, m, 9, 1)
    error1_oja_27 = oja_k(X, m, 27, 1)
    error1_oja_81 = oja_k(X, m, 81, 1)
    error1_oja_243 = oja_k(X, m, 243, 1)

    error_k_pca = vr_pca_k(X, m, eta, 1)
    error_pi_k = power_iterations_k(X, eta, 1)
    error_k_pca_better = vr_pca_better_k(X, m, eta, 1)

    plt.figure(figsize=(7, 7))
    plt.plot(np.arange(1, len(error1_oja_1) + 1), error1_oja_1, label='oja eta=1')
    plt.plot(np.arange(1, len(error1_oja_3) + 1), error1_oja_3, label='oja eta=3')
    plt.plot(np.arange(1, len(error1_oja_9) + 1), error1_oja_9, label='oja eta=9')
    plt.plot(np.arange(1, len(error1_oja_27) + 1), error1_oja_27, label='oja eta=27')
    plt.plot(np.arange(1, len(error1_oja_81) + 1), error1_oja_81, label='oja eta=81')
    plt.plot(np.arange(1, len(error1_oja_243) + 1), error1_oja_243, label='oja eta=243')
    plt.plot(np.arange(1, len(error_pi_k) + 1), error_pi_k, label='power iteration')
    plt.plot(np.arange(1, len(error_k_pca) + 1), error_k_pca, label='vr-pca-k')
    plt.plot(error_k_pca_better, label='vr pca k better')
    plt.xlabel("# Data iterations")
    plt.ylabel('Loss')
    plt.ylim([-10, 0])
    plt.title("Convergence of different methods for k=1")
    plt.legend()


def create_graphs_for_iris_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
    data_pd = df.drop(columns=["target"], axis=1)

    X = data_pd.to_numpy().T
    n = 150
    m = n
    r_h = (X ** 2).sum() / n
    eta = 1 / (r_h * np.sqrt(n))

    error_k_pca = vr_pca_k(X, m, eta, 1)
    error_pi_k = power_iterations_k(X, eta, 1)
    error1_oja_1 = oja_k(X, m, 1, 1)
    error1_oja_3 = oja_k(X, m, 3, 1)
    error1_oja_9 = oja_k(X, m, 9, 1)
    error1_oja_27 = oja_k(X, m, 27, 1)
    error1_oja_81 = oja_k(X, m, 81, 1)
    error1_oja_243 = oja_k(X, m, 243, 1)

    plt.figure(figsize=(7, 7))
    plt.plot(np.arange(1, len(error1_oja_1) + 1), error1_oja_1, label='oja eta=1')
    plt.plot(np.arange(1, len(error1_oja_3) + 1), error1_oja_3, label='oja eta=3')
    plt.plot(np.arange(1, len(error1_oja_9) + 1), error1_oja_9, label='oja eta=9')
    plt.plot(np.arange(1, len(error1_oja_27) + 1), error1_oja_27, label='oja eta=27')
    plt.plot(np.arange(1, len(error1_oja_81) + 1), error1_oja_81, label='oja eta=81')
    plt.plot(np.arange(1, len(error1_oja_243) + 1), error1_oja_243, label='oja eta=243')
    plt.plot(np.arange(1, len(error_pi_k) + 1), error_pi_k, label='power iteration')
    plt.plot(np.arange(1, len(error_k_pca) + 1), error_k_pca, label='vr-pca-k')
    plt.xlabel("# Data iterations")
    plt.ylabel('Loss')
    plt.ylim([-10, 0])
    plt.title("IRIS dataset - Convergence of different methods for k=1")
    plt.legend()


if __name__ == '__main__':
    print('\n######################################################################################################\n')
    print("Please select a function to execute:")
    print("1. Run a comparison between Oja, power iteration and vr_pca algorithms for k=1 on mnist dataset.")
    print("2. Run a comparison between Oja, power iteration and vr_pca algorithms for k=6 on mnist dataset.")
    print("3. Run the improved vr_pca algorithm on mnist dataset compared to Oja, power iteration and vr_pca.")
    print("4. Run Oja, power iteration, vr_pca and the improved version of vr_pca on iris dataset.")
    print('\n######################################################################################################\n')

    while True:
        try:
            choice = int(input("Enter a number (1-4): "))
            if 1 <= choice <= 4:
                break
            else:
                print("Invalid input. Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")
    
    if choice == 1:
        create_graphs_for_mnist_dataset_k(k=1)
    elif choice == 2:
        create_graphs_for_mnist_dataset_k(k=6)
    elif choice == 3:
        create_graphs_for_improved_algorithm()
    elif choice == 4:
        create_graphs_for_iris_dataset()


    plt.show()
