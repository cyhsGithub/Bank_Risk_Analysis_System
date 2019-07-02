import pandas as pd

class proc:
    def __init__(self,data):
        self.data = data

    def normalization(self, columns):
        self.data = self.data.apply(lambda x: (x - x.mean()) / x.std() if x.name in columns else x)
        return self.data

    def discretization(self, columns):
        self.data = self.data.apply(lambda x: pd.qcut(x,q=5,duplicates='drop') if x.name in columns else x)
        self.data = pd.get_dummies(self.data)
        return self.data



