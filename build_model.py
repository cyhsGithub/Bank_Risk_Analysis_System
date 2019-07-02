from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class build_model:
    def __init__(self,data):
        self.data = data

    def LR_build(self, penalty = 'l2', C = 1.0, solver='liblinear'):
        # 筛选data中的Default列的值，赋予变量y
        y = self.data['Default'].values

        # 筛选除去Default列的其他列的值，赋予变量x
        x = self.data.drop(['Default'], axis=1).values

        # 使用train_test_split方法，将x,y划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, stratify=y)

        # 调用模型，新建模型对象
        lr = LogisticRegression(penalty=penalty,C=C, solver=solver)

        # 带入训练集x_train, y_train进行训练
        lr.fit(x_train, y_train)

        # 对训练好的lr模型调用predict方法,带入测试集x_test进行预测
        # y_predict = lr.predict(x_test)

        y_predict_proba = lr.predict_proba(x_test)

        # 取目标分数为正类(1)的概率估计
        y_predict = y_predict_proba[:, 1]

        # 利用precision_recall curve查看模型效果
        test_auc = roc_auc_score(y_true=y_test, y_score=y_predict)

        # 利用roc_auc_score查看模型效果
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
        plt.plot(precision, recall)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.show()

        F1_score = 2 * precision * recall / (precision + recall)
        print("F-score of LR: ", max(F1_score)) #对于不同的precision 和recall 的组合，选出F-score最大的

        return lr, test_auc

    def RF_build(self, n=10, max_depth=None):
        # 筛选data中的Default列的值，赋予变量y
        y = self.data['Default'].values

        # 筛选除去Default列的其他列的值，赋予变量x
        x = self.data.drop(['Default'], axis=1).values

        # 使用train_test_split方法，将x,y划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, stratify=y)


        rf_clf = RandomForestClassifier(n_estimators=n, max_depth=max_depth)
        rf_clf.fit(x_train, y_train)
        y_predict = rf_clf.predict_proba(x_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_predict)

        # 利用roc_auc_score查看模型效果
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
        plt.plot(precision, recall)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.show()

        F1_score = 2 * precision * recall / (precision + recall)
        print("F-score of RF: ", max(F1_score)) #对于不同的precision 和recall 的组合，选出F-score最大的

        return rf_clf, test_auc

    def svm_build(self):
        # 筛选data中的Default列的值，赋予变量y
        y = self.data['Default'].values

        # 筛选除去Default列的其他列的值，赋予变量x
        x = self.data.drop(['Default'], axis=1).values

        # 使用train_test_split方法，将x,y划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, stratify=y)

        parameters = {'kernel': ('linear', 'rbf'), 'C': [1,5]}

        svm_clf = SVC()
        clf = GridSearchCV(svm_clf, parameters, n_jobs=4)
        clf.fit(x_train, y_train)

        y_predict_proba = clf._predict_proba_lr(x_test)
        y_predict = y_predict_proba[:,1]

        test_auc = roc_auc_score(y_true=y_test, y_score=y_predict)

        return clf, test_auc

    def knn_build(self):
        # 筛选data中的Default列的值，赋予变量y
        y = self.data['Default'].values

        # 筛选除去Default列的其他列的值，赋予变量x
        x = self.data.drop(['Default'], axis=1).values

        # 使用train_test_split方法，将x,y划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, stratify=y)

        neighbors = []
        test_score = []

        #使用默认参数
        # clf = None
        # knn_clf = KNeighborsClassifier()
        # knn_clf.fit(x_train, y_train)
        # y_predict = knn_clf.predict_proba(x_test)[:,1]
        # test_auc = roc_auc_score(y_test, y_predict)
        #
        # precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
        # plt.plot(precision, recall)
        # plt.xlabel('precision')
        # plt.ylabel('recall')
        # plt.show()
        #
        # F1_score = 2 * precision * recall / (precision + recall)
        # print("F-score of knn: ", max(F1_score))

        #寻找最优参数区间
        # for k in range(30, 55, 2):
        #     neighbors.append(k)
        #     knn_clf = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree')
        #     knn_clf.fit(x_train, y_train)
        #     y_predict = knn_clf.predict_proba(x_test)[:,1]
        #     test_score.append(roc_auc_score(y_test,y_predict))
        #
        # clf = None
        # test_auc = max(test_score)
        # plt.xlabel('n_neighbors')
        # plt.ylabel('AUC')
        # plt.plot(neighbors, test_score, label='test set')
        # plt.show()

        #GridSearch
        knn_clf = KNeighborsClassifier(algorithm='kd_tree')
        parameters = {'n_neighbors': [32, 37],'weights': ('uniform', 'distance')}

        clf = GridSearchCV(knn_clf, parameters, cv=5, n_jobs=4)
        clf.fit(x_train, y_train)

        y_predict_proba = clf.predict_proba(x_test)
        y_predict = y_predict_proba[:, 1]

        test_auc = roc_auc_score(y_true=y_test, y_score=y_predict)


        return clf, test_auc



