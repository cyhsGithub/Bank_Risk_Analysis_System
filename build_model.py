from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

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

        # 利用roc_auc_score查看模型效果
        test_auc = roc_auc_score(y_true=y_test, y_score=y_predict)

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

        return rf_clf, test_auc

    def svm_build(self):
        # 筛选data中的Default列的值，赋予变量y
        y = self.data['Default'].values

        # 筛选除去Default列的其他列的值，赋予变量x
        x = self.data.drop(['Default'], axis=1).values

        # 使用train_test_split方法，将x,y划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, stratify=y)

        parameters = {'kernel': ('linear', 'rbf'), 'C': [1,10]}

        svm_clf = SVC()
        clf = GridSearchCV(svm_clf, parameters, cv=5)
        clf.fit(x_train, y_train)

        y_predict_proba = clf._predict_proba_lr(x_test)
        y_predict = y_predict_proba[:,1]

        test_auc = roc_auc_score(y_true=y_test, y_score=y_predict)

        return clf, test_auc



