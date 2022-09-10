# import pandas
# import xgboost as xgb
# from xgboost import plot_importance
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score   # 准确率
#
#
# data_frame = pandas.read_csv(
#     './dataset/exampleForLUAD.csv', header=0, index_col=0,
#     encoding='utf-8',
#     skip_blank_lines=True
# )
#
# x = data_frame.iloc[:, :-1].values
# y = data_frame.iloc[:, -1].values
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2) # 数据集分割
#
# # 算法参数
# params = {
#     'booster': 'gbtree',
#     'objective': 'multi:softmax',
#     'num_class': 3,
#     'gamma': 0.1,
#     'max_depth': 6,
#     'lambda': 2,
#     'subsample': 0.7,
#     'colsample_bytree': 0.75,
#     'min_child_weight': 3,
#     'silent': 0,
#     'eta': 0.1,
#     'seed': 1,
#     'nthread': 4,
# }
#
# plst = list(params.items())
#
# dtrain = xgb.DMatrix(x_train, y_train) # 生成数据集格式
# num_rounds = 500
# model = xgb.train(plst, dtrain, num_rounds) # xgboost模型训练
#
# # 对测试集进行预测
# dtest = xgb.DMatrix(x_test)
# y_pred = model.predict(dtest)
#
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print("accuarcy: %.2f%%" % (accuracy*100.0))
#
# # 显示重要特征
# plot_importance(model)
# plt.show()

import pandas
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy

data_frame = pandas.read_csv(
    './dataset/exampleForLUAD.csv', header=0, index_col=0,
    encoding='utf-8',
    skip_blank_lines=True
)

x = data_frame.iloc[:, :-1].values
y = data_frame.iloc[:, -1].values

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=1)   # 分训练集和验证集
parameters = {
              'max_depth': [5, 10, 15, 20, 25],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'n_estimators': [500, 1000, 2000, 3000, 5000],
              'min_child_weight': [0, 2, 5, 10, 20],
              'max_delta_step': [0, 0.2, 0.6, 1, 2],
              'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
              'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
              'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
}

xlf = xgb.XGBClassifier(max_depth=10,
            learning_rate=0.01,
            n_estimators=2000,
            silent=True,
            objective='multi:softmax',
            num_class=3 ,
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=0.85,
            colsample_bytree=0.7,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=0)

gs = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
gs.fit(x_train, y_train)

print("Best score: %0.3f" % gs.best_score_)
print("Best parameters set: %s" % gs.best_params_)

