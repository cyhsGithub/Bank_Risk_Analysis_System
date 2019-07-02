import pandas as pd
import numpy as np

class Preprocess:
    def __init__(self, data, one_hot=True):
        self.data = data
        self.one_hot = one_hot
    
    def process(self):
        # 缺失值处理
        self.data = self.data[~self.data['Default'].isin(['nan'])]
        self.data.fillna(value='未知', inplace=True)

        # 异常值处理
        mode = self.data['isCrime'].mode()
        self.data.isCrime[self.data['isCrime'] == 2] = int(mode)

        # 将网上消费笔数为0时的网上消费金额皆修改为0
        self.data.onlineTransAmt[self.data['onlineTransCnt'] == 0] = 0

        # 将公共事业缴费笔数为0时的公共事业缴费金额皆修改为0
        self.data.publicPayAmt[self.data['publicPayCnt'] == 0] = 0

        # 从self.data中筛选总消费笔数小于6000的值，赋值给self.data
        self.data = self.data[self.data['transTotalCnt'] < 6000]

        # 数字编码
        self.data["maritalStatus"] = self.data["maritalStatus"].map({"未知": 0, "未婚": 1, "已婚": 2})
        self.data['education'] = self.data["education"].map({"未知": 0, "小学": 1, "初中": 2, "高中": 3, "本科以上": 4})
        self.data['idVerify'] = self.data["idVerify"].map({"未知": 0, "一致": 1, "不一致": 2})
        self.data['threeVerify'] = self.data["threeVerify"].map({"未知": 0, "一致": 1, "不一致": 2})
        self.data["netLength"] = self.data["netLength"].map({"无效": 0, "0-6个月": 1, "6-12个月": 2, "12-24个月": 3, "24个月以上": 4})
        self.data["sex"] = self.data["sex"].map({"未知": 0, "男": 1, "女": 2})
        self.data["CityId"] = self.data["CityId"].map({"一线城市": 1, "二线城市": 2, "其它": 3})

        # One-Hot 编码
        if self.one_hot:
            columns = ['maritalStatus', 'education', 'idVerify', 'threeVerify', 'Han', 'netLength', 'sex', 'CityId']
            self.data = pd.get_dummies(data=self.data, columns=columns)

        # 新指标计算
        # 计算客户年消费总额。
        trans_total = self.data['transCnt_mean'] * self.data['transAmt_mean']

        # 将计算结果保留到小数点后六位。
        trans_total = trans_total.round(6)

        # 将结果加在self.data数据集中的最后一列，并将此列命名为trans_total。
        self.data['trans_total'] = trans_total

        # 计算客户年取现总额。
        total_withdraw = self.data['cashCnt_mean'] * self.data['cashAmt_mean']

        # 将计算结果保留到小数点后六位。
        total_withdraw = total_withdraw.round(6)

        # 将结果加在self.data数据集的最后一列，并将此列命名为total_withdraw。
        self.data['total_withdraw'] = total_withdraw

        # 计算客户的平均每笔取现金额。
        avg_per_withdraw = self.data['cashTotalAmt'] / self.data['cashTotalCnt']

        # 将所有的inf和NaN变为0。
        avg_per_withdraw = avg_per_withdraw.replace([np.inf, np.nan], 0)

        # 将计算结果保留到小数点后六位。
        avg_per_withdraw = avg_per_withdraw.round(6)

        # 将结果加在self.data数据集的最后一列，并将此列命名为avg_per_withdraw。
        self.data['avg_per_withdraw'] = avg_per_withdraw

        # 计算客户的网上平均每笔消费额。
        avg_per_online_spend = self.data['onlineTransAmt'] / self.data['onlineTransCnt']

        # 将所有的inf和NaN变为0。
        avg_per_online_spend = avg_per_online_spend.replace([np.inf, np.nan], 0)

        # 将计算结果保留到小数点后六位。
        avg_per_online_spend = avg_per_online_spend.round(6)

        # 将结果加在self.data数据集的最后一列，并将此列命名为avg_per_online_spend。
        self.data['avg_per_online_spend'] = avg_per_online_spend

        # 计算客户的公共事业平均每笔缴费额。
        avg_per_public_spend = self.data['publicPayAmt'] / self.data['publicPayCnt']

        # 将所有的inf和NaN变为0。
        avg_per_public_spend = avg_per_public_spend.replace([np.inf, np.nan], 0)

        # 将计算结果保留到小数点后六位。
        avg_per_public_spend = avg_per_public_spend.round(6)

        # 将结果加在self.data数据集的最后一列，并将此列命名为avg_per_public_spend。
        self.data['avg_per_public_spend'] = avg_per_public_spend

        # 计算客户的不良记录分数。
        bad_record = self.data['inCourt'] + self.data['isDue'] + self.data['isCrime'] + self.data['isBlackList']

        # 将计算结果加在self.data数据集的最后一列，并将此列命名为bad_record。
        self.data['bad_record'] = bad_record

        return self.data





