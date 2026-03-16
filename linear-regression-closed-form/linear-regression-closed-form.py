import numpy as np

def linear_regression_closed_form(X, y):
    """
    X shape: (N, features)
    y shape: (N,)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Bước 1: Tính X^T * X
    # Kết quả là ma trận (features, features)
    XTX = X.T @ X
    
    # Bước 2: Thêm Regularization (Ridge) để tăng tính ổn định
    # Giúp tránh lỗi Singular Matrix nếu các features bị trùng lặp
    XTX_stable = XTX + np.eye(XTX.shape[0]) * 1e-5
    
    # Bước 3: Tính w = (XTX)^-1 * X^T * y
    # Thay vì dùng inv, dùng pinv sẽ an toàn hơn nữa
    w = np.linalg.inv(XTX_stable) @ X.T @ y
    
    return w