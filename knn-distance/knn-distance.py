import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # ---------------------------------------------------------
    # BƯỚC BẢO VỆ CỔNG (EARLY EXIT): Xử lý ngay nếu X_test rỗng
    if len(X_test) == 0:
        # np.empty tạo ra ma trận rỗng cực nhanh với shape (0, k)
        return np.empty((0, k), dtype=int) 
    # ---------------------------------------------------------
    
    result = []
    for x_test in X_test:
        distances = []
        for x_train in X_train:
            distances.append(np.linalg.norm(x_test - x_train))
            
        distances = np.array(distances)
        sorted_idx = np.argsort(distances)
        result.append(sorted_idx[:k]) 
        
    result = np.array(result)
    
    # Xử lý ngoại lệ: k lớn hơn số lượng điểm train
    if k > len(X_train):
        hang = len(X_test)
        cot = k - len(X_train)
        them = np.full((hang, cot), -1) 
        result = np.hstack((result, them))
        
    return result