'''
参数调整注意事项
1.控制过拟合：
  a.控制模型复杂度max_depth, min_child_weight和gamma
  b.增加随机性。sub_sample, colsample_bytree减小步长eta,但同时要增加num_round
2.pass

参数详解
一.常规参数
1.booster[default = gbtree]
选择基分类器，gbtree:基于树的模型，gblinear:线性模型
2.silent [default=0]
0输出运行信息，1不输出
3.nthread [default to maximum number of threads available if not set]
线程数
4.num_pbuffer [set automatically by xgboost, no need to be set by user]
5.num_feature [set automatically by xgboost, no need to be set by user]
二.模型参数Booster Parameters
1.eta [default=0.3]
shrinkage参数，防止过拟合。用于更新叶子节点权重时，乘以该系数，避免步长过大。
参数太大可能不收敛。range: [0,1]
2.gamma [default=0]
后剪枝时，用于控制是否剪枝 range: [0,∞]
3.max_depth [default=6]
每棵树的最大深度，树高越深，越容易过拟合 range: [1,∞]
4.min_child_weight [default=1]
minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
range: [0,∞]
5.max_delta_step [default=0]
Maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
range: [0,∞]
6.subsample [default=1]
采样率，防止过拟合。 Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
range: (0,1]
7.colsample_bytree [default=1]
列采样率，对每棵树的生成用的特征进行列采样 range: (0,1]
8.colsample_bylevel [default=1]
每次分裂的列采样subsample ratio of columns for each split, in each level.
range: (0,1]
9.lambda [default=1]
控制模型复杂度的权重的L2 正则化参数，越大越不容易过拟合
10.alpha [default=0]
控制模型复杂度的权重的L1 正则化参数，越大越不容易过拟合
11.tree_method, string [default=’auto’]
The tree constructtion algorithm used in XGBoost(see description in the reference paper)
Distributed and external memory version only support approximate algorithm.
Choices: {‘auto’, ‘exact’, ‘approx’}
‘auto’: Use heuristic to choose faster one.
For small to medium dataset, exact greedy will be used.
For very large-dataset, approximate algorithm will be choosed.
Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is choosed to notify this choice.
‘exact’: Exact greedy algorithm.
‘approx’: Approximate greedy algorithm using sketching and histogram.
12.sketch_eps, [default=0.03]
This is only used for approximate greedy algorithm.
This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical ganrantee with sketch accuracy.
Usuaully user do not have to tune this. but consider set to lower number for more accurate enumeration.
range: (0, 1)
13.scale_pos_weight, [default=0]
控制正负样本权重平衡Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases) See Parameters Tuning for more discussion. Also see Higgs Kaggle competition demo for examples: R, py1, py2, py3
三.学习任务的参数
1.objective [ default=reg:linear ]
“reg:linear” –linear regression
“reg:logistic” –logistic regression
“binary:logistic” –logistic regression for binary classification, output probability
“binary:logitraw” –logistic regression for binary classification, output score before logistic transformation
“count:poisson” –poisson regression for count data, output mean of poisson distribution
max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
“multi:softmax” –set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
“multi:softprob” –same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
“rank:pairwise” –set XGBoost to do ranking task by minimizing the pairwise loss
“reg:gamma” –gamma regression for severity data, output mean of gamma distribution
2.base_score [ default=0.5 ]
the initial prediction score of all instances, global bias
for sufficent number of iterations, changing this value will not have too much effect.
3.eval_metric [ default according to objective ]
evaluation metrics for validation data, a default metric will be assigned according to objective( rmse for regression, and error for classification, mean average precision for ranking )
User can add multiple evaluation metrics, for python user, remember to pass the metrics in as list of parameters pairs instead of map, so that latter ‘eval_metric’ won’t override previous one
The choices are listed below:
“rmse”: root mean square error
“mae”: mean absolute error
“logloss”: negative log-likelihood
“error”: Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
“merror”: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
“mlogloss”: Multiclass logloss
“auc”: Area under the curve for ranking evaluation.
“ndcg”:Normalized Discounted Cumulative Gain
“map”:Mean average precision
“ndcg@n”,”map@n”: n can be assigned as an integer to cut off the top positions in the lists for evaluation.
“ndcg-”,”map-”,”ndcg@n-”,”map@n-”: In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding “-” in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions. training repeatively
“gamma-deviance”: [residual deviance for gamma regression]
4.seed [ default=0 ]




import xgboost as xgb
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
param = {'max_depth': 2, 'eta': 1,'objective':'binary:logistic'}
num_round = 2
bst = xdb.train(param, dtrain, num_round)
preds = bst.predict(detst)

import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'booster': 'dart',
         'max_depth': 5, 'learning_rate': 0.1,
         'objective': 'binary:logistic', 'silent': True,
         'sample_type': 'uniform',#采样算法的类型。uniform(默认）：drop的树被统一选择，weighted：根据weights选择drop的树
         'normalize_type': 'tree',#归一化算法的类型，tree(默认）：新树与drop的树weight相同。forest:新树与drop的树的权重总和相同
         'rate_drop': 0.1,
         'skip_drop': 0.5}
num_round = 50
bst = xgb.train(param, dtrain, num_round)
# make prediction
# ntree_limit must not be 0
preds = bst.predict(dtest, ntree_limit=num_round)


xgb_model = xgb.XGBRessor(max_depth = 3, learning_rate = 0.1, n_estimators = 100, objective = 'reg:linear', njobs = -1)
xgb_model.fit(X_train, y_train, eval_set = [(X_train, y_train)], eval_metric = 'logloss', verbose = 100)
y_pred = xgb_model.predict(X_test)
print(mean_square_error(y_test, y_pred))
