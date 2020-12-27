Step1:定义数据预处理的步骤
  数值型数据缺失值
  分类型数据缺失值和one-hot编码
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprossing import OneHotEncoder
#数值型数据的处理
numerical_transformer = SimpleImputer(strategy='constant')
#分类型数据的处理
categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHoTEconder(handle_unknown='ignore'))
])
#把两种处理捆绑
preprocessor = ColumnTransformer(
  transformers=[
  ('num', numerical_transformer, numerical_cols),
  ('cat', categorical_transformer, categorical_cols)
])
step2:定义模型
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)
step3:创建并评估Pipeline
from sklearn.metrics import mean_absolute_error
#捆绑预处理和模型
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
#预处理训练数据并拟合
my_pipeline.fit(X_train, y_train)
#处理验证集并预测
preds = my_pipeline.predict(X_valid)
#评估模型
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
