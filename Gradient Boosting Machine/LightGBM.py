# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, accuracy_score
#
# # 加载数据
# data_frame = pd.read_csv(
#     './dataset/exampleForLUAD.csv', header=0, index_col=0,
#     encoding='utf-8',
#     skip_blank_lines=True
# )
#
# # 划分训练集和测试集
# x = data_frame.iloc[:, :-1].values
# y = data_frame.iloc[:, -1].values
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# # 转换为Dataset数据格式
# train_data = lgb.Dataset(x_train, label=y_train)
# validation_data = lgb.Dataset(x_test, label=y_test)
#
# # 参数
# params = {
#     'learning_rate': 0.1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.2,
#     'max_depth': 4,
#     'objective': 'multiclass',  # 目标函数
#     'num_class': 3,
# }
#
# # 模型训练
# gbm = lgb.train(params, train_data, valid_sets=[validation_data])
#
# # 模型预测
# y_pred = gbm.predict(x_test)
# y_pred = [list(x).index(max(x)) for x in y_pred]
# print(y_pred)
#
# # 模型评估
# print(accuracy_score(y_test, y_pred))

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

# 加载数据
iris = load_iris()
data = iris.data
target = iris.target

# 划分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 模型训练
gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=5)

# 模型存储
joblib.dump(gbm, 'loan_model.pkl')
# 模型加载
gbm = joblib.load('loan_model.pkl')

# 模型预测
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)

# 模型评估
print('The accuracy of prediction is:', accuracy_score(y_test, y_pred))

# 特征重要度
print('Feature importances:', list(gbm.feature_importances_))

# 网格搜索，参数优化
estimator = LGBMClassifier(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(x_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)