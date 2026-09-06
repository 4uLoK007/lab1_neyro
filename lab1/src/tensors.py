"""Задания 38-45: тензорные вычисления TensorFlow, PyTorch, Keras."""

import numpy as np
import tensorflow as tf
import torch
import keras
from keras import ops


# =====================================================================
# Задание 38. Тензор TensorFlow
# =====================================================================
def make_tensor_tf(data: np.ndarray):
    """Создаёт tf.Tensor из numpy-массива."""
    return tf.constant(data, dtype=tf.float32)


# =====================================================================
# Задание 39. Тензор PyTorch
# =====================================================================
def make_tensor_pt(data: np.ndarray):
    """Создаёт torch.Tensor из numpy-массива."""
    return torch.tensor(data, dtype=torch.float32)


# =====================================================================
# Задание 40. Случайные тензоры
# =====================================================================
def make_random_tensors(n: int, m: int, k: int = 5, p: int = 4):
    """Создаёт случайные A, W, B для операции A @ X @ W + B.

    A: (k, n) — целочисленный
    W: (m, p) — нормальный
    B: (k, p) — равномерный
    """
    A = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
    W = tf.random.normal((m, p))
    B = tf.random.uniform((k, p))
    return A, W, B


# =====================================================================
# Задание 41. Операция A @ X @ W + B (TensorFlow)
# =====================================================================
def axwb_tensorflow(X_tf, A, W, B):
    """AXW + B через TensorFlow."""
    AX = tf.matmul(tf.cast(A, tf.float32), X_tf)
    AXW = tf.matmul(AX, W)
    return AXW + B


# =====================================================================
# Задание 42. Операция A @ X @ W + B (PyTorch)
# =====================================================================
def axwb_pytorch(X_pt, A, W, B):
    """AXW + B через PyTorch."""
    A_pt = torch.tensor(A.numpy(), dtype=torch.float32)
    W_pt = torch.tensor(W.numpy(), dtype=torch.float32)
    B_pt = torch.tensor(B.numpy(), dtype=torch.float32)
    return A_pt @ X_pt @ W_pt + B_pt


# =====================================================================
# Задание 43. Матрицы T, P, Q
# =====================================================================
def make_TPQ(shape=(3, 10)) -> tuple:
    """Случайные T, P, Q формы (3, 10)."""
    T = np.random.rand(*shape)
    P = np.random.rand(*shape)
    Q = np.random.rand(*shape)
    return T, P, Q


# =====================================================================
# Задание 44. Операция в Keras
# =====================================================================
def compute_V_keras(T, P, Q):
    """V = |sin(T) - exp(P) * sqrt(Q)| через Keras ops."""
    T_k = ops.convert_to_tensor(T, dtype="float32")
    P_k = ops.convert_to_tensor(P, dtype="float32")
    Q_k = ops.convert_to_tensor(Q, dtype="float32")
    V_k = ops.abs(ops.sin(T_k) - ops.exp(P_k) * ops.sqrt(Q_k))
    return V_k


# =====================================================================
# Задание 45. Та же операция в PyTorch
# =====================================================================
def compute_V_pytorch(T, P, Q):
    """V = |sin(T) - exp(P) * sqrt(Q)| через PyTorch."""
    T_t = torch.tensor(T, dtype=torch.float32)
    P_t = torch.tensor(P, dtype=torch.float32)
    Q_t = torch.tensor(Q, dtype=torch.float32)
    V_t = torch.abs(torch.sin(T_t) - torch.exp(P_t) * torch.sqrt(Q_t))
    return V_t


# =====================================================================
# ОБЩАЯ ФУНКЦИЯ ЗАПУСКА 38-45
# =====================================================================
def run_all(data: np.ndarray) -> None:
    """Прогоняет задания 38-45 и печатает результаты."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЯ 38-45: тензоры")
    print("=" * 60)

    n, m = data.shape
    print(f"[38-40] data.shape = ({n}, {m})")

    # --- 38 ---
    X_tf = make_tensor_tf(data)
    print(f"[38] X_tf: shape={X_tf.shape}, dtype={X_tf.dtype.name}")

    # --- 39 ---
    X_pt = make_tensor_pt(data)
    print(f"[39] X_pt: shape={tuple(X_pt.shape)}, dtype={X_pt.dtype}")

    # --- 40 ---
    k, p = 5, 4
    A, W, B = make_random_tensors(n, m, k=k, p=p)
    print(f"[40] A: {A.shape}, W: {W.shape}, B: {B.shape}")

    # --- 41 ---
    result_tf = axwb_tensorflow(X_tf, A, W, B)
    print(f"[41] TF result shape = {tuple(result_tf.shape)}")

    # --- 42 ---
    result_pt = axwb_pytorch(X_pt, A, W, B)
    print(f"[42] PT result shape = {tuple(result_pt.shape)}")
    # Сравним значения (на CPU они должны почти совпадать)
    diff = np.abs(result_tf.numpy() - result_pt.numpy()).max()
    print(f"     max|TF - PT| = {diff:.6e}")

    # --- 43 ---
    T, P, Q = make_TPQ((3, 10))
    print(f"[43] T: {T.shape}, P: {P.shape}, Q: {Q.shape}")

    # --- 44 ---
    V_k = compute_V_keras(T, P, Q)
    print(f"[44] Keras V shape = {tuple(V_k.shape)}")

    # --- 45 ---
    V_t = compute_V_pytorch(T, P, Q)
    print(f"[45] PyTorch V shape = {tuple(V_t.shape)}")

    print("=" * 60)
    print("ЗАДАНИЯ 38-45 ВЫПОЛНЕНЫ")
    print("=" * 60)