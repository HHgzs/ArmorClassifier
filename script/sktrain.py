from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from script.load import *

def train_svm(X_train, y_train):
    """
    使用 SVM 训练分类器。
    
    参数:
    X_train (numpy.ndarray): 训练集图像。
    y_train (numpy.ndarray): 训练集标签。
    
    返回:
    svm_clf: 训练好的 SVM 分类器。
    """
    svm_clf = SVC(kernel='linear', C=1.0, random_state=42)
    svm_clf.fit(X_train, y_train)
    return svm_clf

def train_decision_tree(X_train, y_train):
    """
    使用决策树训练分类器。
    
    参数:
    X_train (numpy.ndarray): 训练集图像。
    y_train (numpy.ndarray): 训练集标签。
    
    返回:
    dt_clf: 训练好的决策树分类器。
    """
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    return dt_clf

def train_logistic_regression(X_train, y_train):
    """
    使用逻辑回归训练分类器。
    
    参数:
    X_train (numpy.ndarray): 训练集图像。
    y_train (numpy.ndarray): 训练集标签。
    
    返回:
    lr_clf: 训练好的逻辑回归分类器。
    """
    lr_clf = LogisticRegression(random_state=42, max_iter=500)
    lr_clf.fit(X_train, y_train)
    return lr_clf

def train_knn(X_train, y_train):
    """
    使用 KNN 训练分类器。
    
    参数:
    X_train (numpy.ndarray): 训练集图像。
    y_train (numpy.ndarray): 训练集标签。
    
    返回:
    knn_clf: 训练好的 KNN 分类器。
    """
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    return knn_clf

def train_random_forest(X_train, y_train):
    """
    使用随机森林训练分类器。
    
    参数:
    X_train (numpy.ndarray): 训练集图像。
    y_train (numpy.ndarray): 训练集标签。
    
    返回:
    rf_clf: 训练好的随机森林分类器。
    """
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    return rf_clf

def train_mlp(X_train, y_train):
    """
    使用多层感知机训练分类器。
    
    参数:
    X_train (numpy.ndarray): 训练集图像。
    y_train (numpy.ndarray): 训练集标签。
    
    返回:
    mlp_clf: 训练好的多层感知机分类器。
    """
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    mlp_clf.fit(X_train, y_train)
    return mlp_clf



def evaluate_model(model, X_test, y_test):
    """
    评估模型在测试集上的性能。
    
    参数:
    model: 训练好的分类器模型。
    X_test (numpy.ndarray): 测试集图像。
    y_test (numpy.ndarray): 测试集标签。
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4%}")
    print("分类报告:")
    print(classification_report(y_test, y_pred))

def main():
    # 测试数据集路径
    train_data_dir = '/home/nvidia/Code/ArmorClassifier/datasets/binary/train'
    test_data_dir = '/home/nvidia/Code/ArmorClassifier/datasets/binary/test'
    
    # 加载数据集
    X, y = load_binary_dataset(train_data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练 SVM 分类器
    print("SVM 分类器: ")
    svm_clf = train_svm(X_train, y_train)
    evaluate_model(svm_clf, X_test, y_test)
    
    # 训练决策树分类器
    print("\n决策树分类器: ")
    dt_clf = train_decision_tree(X_train, y_train)
    evaluate_model(dt_clf, X_test, y_test)

    # 训练逻辑回归分类器
    print("\n逻辑回归分类器: ")
    lr_clf = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_clf, X_test, y_test)

    # 训练 KNN 分类器
    print("\nKNN 分类器: ")
    knn_clf = train_knn(X_train, y_train)
    evaluate_model(knn_clf, X_test, y_test)

    # 训练随机森林分类器
    print("\n随机森林分类器: ")
    rf_clf = train_random_forest(X_train, y_train)
    evaluate_model(rf_clf, X_test, y_test)

    # 训练多层感知机分类器
    print("\n多层感知机分类器: ")
    mlp_clf = train_mlp(X_train, y_train)
    evaluate_model(mlp_clf, X_test, y_test)


if __name__ == '__main__':
    main()