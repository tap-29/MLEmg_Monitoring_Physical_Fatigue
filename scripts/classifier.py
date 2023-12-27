import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class RotationalForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.pca_transformers_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.estimators_ = [RandomForestClassifier(n_estimators=1, random_state=np.random.randint(1e6)) for _ in
                            range(self.n_estimators)]
        self.pca_transformers_ = [PCA(n_components=n_features) for _ in range(self.n_estimators)]

        for i, estimator in enumerate(self.estimators_):
            subset_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X[subset_indices, :]
            y_subset = y[subset_indices]

            X_transformed = self.pca_transformers_[i].fit_transform(X_subset)
            estimator.fit(X_transformed, y_subset)

        return self

    def predict(self, X):
        predictions = []

        for i, estimator in enumerate(self.estimators_):
            X_transformed = self.pca_transformers_[i].transform(X)
            predictions.append(estimator.predict(X_transformed))

        predictions = np.array(predictions, dtype=int)  # 将预测结果转换为整数
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)


def initial_data():
    # 要保证：X的数据格式最后为：样本数*特征数

    # 加载 npy 文件
    fatigue_data = np.load("../data/fatigue_data.npy")
    non_fatigue_data = np.load("../data/non_fatigue_data.npy")

    # 合并两个数据集
    X = np.concatenate((fatigue_data, non_fatigue_data), axis=1)  # 结果矩阵的大小为 8 x (156 + 156)

    # 创建标签向量
    fatigue_labels = np.ones((1, fatigue_data.shape[1]))  # 疲劳样本的标签为 1
    non_fatigue_labels = np.zeros((1, non_fatigue_data.shape[1]))  # 非疲劳样本的标签为 0
    y = np.concatenate((fatigue_labels, non_fatigue_labels), axis=1).reshape(-1)  # 结果向量的大小为 (312,)

    # 例如，将 X 转置为 (312, 8)，这样每行表示一个样本，每列表示一个通道
    X = X.T

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 对特征进行标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # print(X_train.shape) (16080, 8) 数据量*特征数
    # print(y_train.shape) (16080, ) 数据量

    return X_train, y_train, X_test, y_test


def print_result(name, y_test, y_pred):
    print(name)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification report:\n", report)


def classifiers(rotation_forest=False, Bayes=False, SVM_rbf=False, SVM_poly=False):
    dict = {'y_pred_rotation_forest': '', 'y_pred_naive_bayes': '', 'y_pred_rbf': '',
            'y_pred_poly': ''}
    if (rotation_forest):
        # 旋转森林分类器:创建、训练、预测
        rotational_forest_classifier = RotationalForestClassifier(n_estimators=100, random_state=42)
        rotational_forest_classifier.fit(X_train, y_train)
        y_pred_rotation_forest = rotational_forest_classifier.predict(X_test)
        dict['y_pred_rotation_forest'] = y_pred_rotation_forest
    if (Bayes):
        # 高斯朴素贝叶斯分类器
        naive_bayes_classifier = GaussianNB()
        naive_bayes_classifier.fit(X_train, y_train)
        y_pred_naive_bayes = naive_bayes_classifier.predict(X_test)
        dict['y_pred_naive_bayes'] = y_pred_naive_bayes
    if (SVM_rbf):
        # SVM(rbf)分类器
        svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_rbf.fit(X_train, y_train)
        y_pred_rbf = svm_rbf.predict(X_test)
        dict['y_pred_rbf'] = y_pred_rbf

    if (SVM_poly):
        # SVM(poly)分类器
        svm_poly = SVC(kernel='poly', C=1.0, degree=3, gamma='scale')
        svm_poly.fit(X_train, y_train)
        y_pred_poly = svm_poly.predict(X_test)
        dict['y_pred_poly'] = y_pred_poly

    return dict


X_train, y_train, X_test, y_test = initial_data()

res_dict = classifiers(True, True, True, True)

# 输出结果：

print_result("随机森林", y_test, res_dict['y_pred_rotation_forest'])
print_result("高斯朴素贝叶斯", y_test, res_dict['y_pred_naive_bayes'])
print_result("SVM_rbf", y_test, res_dict['y_pred_rbf'])
print_result("SVM_poly", y_test, res_dict['y_pred_poly'])
