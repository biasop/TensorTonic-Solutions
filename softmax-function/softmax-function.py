import numpy as np

def softmax(x):
    x = np.asarray(x)
    
    # Bước 1: Tìm max theo hàng (axis=1) để ổn định số học
    # Dùng keepdims=True để max_x có shape (N, 1), khớp với x (N, K)
    max_x = np.max(x, axis=-1, keepdims=True)
    
    # Bước 2: Tính số mũ (Exponentiate)
    exp_x = np.exp(x - max_x)
    
    # Bước 3: Tính tổng theo hàng (axis=-1) và chia
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)