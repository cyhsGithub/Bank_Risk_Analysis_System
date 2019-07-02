from data_preprocess import Preprocess
from build_model import build_model
from normalization_and_discretization import proc
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

#load data
filename = '.\data.json'
file = open(filename,'r',encoding='utf-8')
all_data = json.load(file)
#convert str_num to num
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

for i in all_data:
    for key in i.keys():
        if i[key]:
            if is_number(i[key]):
                i.update({key: float(i[key])})

data = pd.DataFrame(all_data)
pd.set_option('mode.chained_assignment', None)

#preprocess processor
pre_processor = Preprocess(data)
data = pre_processor.process()

# processor
processor = proc(data)


clf = build_model(data)
# LR model using default penalty = 'l2' and C = 1.0
lr_model, test_auc = clf.LR_build(solver='lbfgs')
print('LR model test_auc, using default penalty = "l2" and C = 1.0: ', test_auc)
# using penalty = 'l1' and C = 0.6
lr_model, test_auc = clf.LR_build(penalty = "l1", C = 0.6)
print('LR model test_auc, using default penalty = "l1" and C = 0.6: ', test_auc)

# apply normalization
continuous_columns = ['age','cashTotalAmt','cashTotalCnt','monthCardLargeAmt','onlineTransAmt','onlineTransCnt','publicPayAmt','publicPayCnt','transTotalAmt','transTotalCnt','transCnt_non_null_months','transAmt_mean','transAmt_non_null_months','cashCnt_mean','cashCnt_non_null_months','cashAmt_mean','cashAmt_non_null_months','card_age', 'trans_total','total_withdraw', 'avg_per_withdraw','avg_per_online_spend', 'avg_per_public_spend', 'bad_record']
new_data = processor.normalization(continuous_columns)
new_clf = build_model(new_data)
lr_model, test_auc = new_clf.LR_build(penalty = "l1", C = 0.6)
print('LR model test_auc, applying normalization: ', test_auc)

# apply discretization
new_data = pd.DataFrame(all_data)
new_pre_processor = Preprocess(data, one_hot=False)
continuous_columns = ['age','cashTotalAmt','cashTotalCnt','monthCardLargeAmt','onlineTransAmt','onlineTransCnt','publicPayAmt','publicPayCnt','transTotalAmt','transTotalCnt','transCnt_non_null_months','transAmt_mean','transAmt_non_null_months','cashCnt_mean','cashCnt_non_null_months','cashAmt_mean','cashAmt_non_null_months','card_age', 'trans_total', 'total_withdraw', 'avg_per_withdraw','avg_per_online_spend', 'avg_per_public_spend', 'bad_record']
new_data = processor.discretization(continuous_columns)
new_clf = build_model(new_data)
lr_model, test_auc = new_clf.LR_build(penalty = "l1", C = 0.6)
print('LR model test_auc, applying normalization', test_auc)

# RF model using default n_estimators=10, max_depth=None
rf_model, test_auc = clf.RF_build()
print('RF model test_auc, using default n_estimators=10, max_depth=None: ', test_auc)
rf_model, test_auc = clf.RF_build(n=100, max_depth=8)
print('RF model test_auc, using n_estimators=100, max_depth=8: ', test_auc)

#KNN model using GridSearch to choose the best parameters "n_neighbors" and "weights"
knn_model, test_auc = clf.knn_build()
print('knn model test_auc, using GridSearch to choose the best parameters "n_neighbors" and "weights": ', test_auc)
