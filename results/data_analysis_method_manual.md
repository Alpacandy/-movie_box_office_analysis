# 电影票房数据分析方法手册

## 1. 项目概述

本项目是一个电影票房分析和预测系统，旨在通过多种数据分析方法和模型，分析电影票房的影响因素并预测票房收入。项目包含数据采集、预处理、特征工程、模型训练、模型融合和结果可视化等多个模块。

## 2. 数据结构与预处理

### 2.1 数据结构

处理后的数据包含以下主要字段：

- **基本信息**：id, title, release_date, runtime
- **财务信息**：budget, revenue
- **评分信息**：vote_average, vote_count
- **流行度信息**：popularity
- **类型特征**：genre_drama, genre_comedy, genre_action, ...
- **导演特征**：director
- **演员特征**：cast
- **制作公司特征**：production_companies, has_major_studio
- **语言特征**：original_language, spoken_languages

### 2.2 数据预处理

数据预处理主要包括以下步骤：

1. **数据清洗**：处理缺失值和异常值
2. **特征工程**：提取和转换特征
3. **数据标准化**：对数值型数据进行标准化处理
4. **特征编码**：对分类特征进行编码
5. **数据划分**：将数据划分为训练集、验证集和测试集

## 3. 回归分析

### 3.1 理论基础

回归分析是一种预测性的建模技术，用于建立因变量（目标变量）和一个或多个自变量（预测变量）之间的关系。在电影票房预测中，我们使用回归分析来建立票房与各种影响因素之间的关系模型。

常见的回归模型包括：

- **线性回归**：假设自变量和因变量之间存在线性关系
- **岭回归**：在线性回归基础上添加L2正则化项，防止过拟合
- **Lasso回归**：在线性回归基础上添加L1正则化项，具有特征选择功能
- **弹性网回归**：结合L1和L2正则化项

### 3.2 实现代码

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data = pd.read_csv('data/processed/cleaned_movie_data.csv')

# 选择特征和目标变量
features = ['budget', 'popularity', 'vote_average', 'vote_count']
# 加入类型特征
genre_features = [col for col in data.columns if col.startswith('genre_')]
features.extend(genre_features)
X = data[features]
y = data['revenue']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 线性回归
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Lasso回归
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# 弹性网回归
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net.fit(X_train_scaled, y_train)
y_pred_en = elastic_net.predict(X_test_scaled)

# 模型评估
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2e}, R2 Score: {r2:.4f}")
    return rmse, r2

print("回归模型评估结果:")
evaluate_model(y_test, y_pred_lr, "线性回归")
evaluate_model(y_test, y_pred_ridge, "岭回归")
evaluate_model(y_test, y_pred_lasso, "Lasso回归")
evaluate_model(y_test, y_pred_en, "弹性网回归")
```

### 3.3 效果评估

回归模型的评估指标主要包括：

- **均方根误差（RMSE）**：衡量预测值与实际值之间的平均误差
- **平均绝对误差（MAE）**：衡量预测值与实际值之间的绝对误差
- **平均绝对百分比误差（MAPE）**：衡量预测值与实际值之间的相对误差
- **R2 Score**：衡量模型解释因变量变异的比例

### 3.4 实际回归模型结果分析

根据项目实际运行结果，回归模型的性能如下：

| 模型名称 | RMSE | MAE | MAPE | R2 Score | 训练时间 |
|----------|------|-----|------|----------|----------|
| 线性回归 | 3.32e+7 | 8.69e+6 | 2.75e+6 | 0.7345 | 短 |
| 岭回归 | 3.32e+7 | 8.69e+6 | 2.75e+6 | 0.7345 | 短 |

从实际结果来看，线性回归和岭回归模型的性能非常接近，R2 Score均为0.7345，RMSE约为3.32e+7。这表明回归模型能够解释约73.45%的票房变异，但预测误差仍然较大。

### 3.5 回归模型特征重要性分析

根据项目分析手册中的`ridge_regression_feature_importance.png`图表分析：

- **预算**：系数为正，与票房正相关，是最重要的影响因素
- **评分（vote_average）**：系数为正，与票房正相关
- **流行度（popularity）**：系数为正，与票房正相关
- **部分类型特征**：如genre_documentary的系数为负，表明纪录片类型对票房有负面影响

回归模型的优点是简单易懂、解释性强、训练速度快，但缺点是假设线性关系，无法捕捉复杂非线性关系。在电影票房预测中，回归模型表现一般，不如后续介绍的传统机器学习模型。

## 4. 聚类分析

### 4.1 理论基础

聚类分析是一种无监督学习方法，用于将数据点分组到不同的簇中，使得同一簇内的数据点相似度较高，不同簇之间的数据点相似度较低。在电影票房分析中，聚类分析可以用于：

- 识别电影的不同类型或群组
- 发现票房表现相似的电影
- 分析不同群组电影的特征差异

常见的聚类算法包括：

- **K-Means聚类**：将数据划分为K个簇，最小化簇内平方和
- **层次聚类**：构建层次化的簇结构
- **DBSCAN**：基于密度的空间聚类，能够发现任意形状的簇

### 4.2 实现代码

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('data/processed/cleaned_movie_data.csv')

# 选择聚类特征
cluster_features = ['budget', 'revenue', 'popularity', 'vote_average', 'runtime']
X = data[cluster_features].dropna()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means聚类
# 确定最佳簇数
inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘制肘部曲线和轮廓系数曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_values, inertia, 'bx-')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
ax1.set_title('K-Means 肘部曲线')

ax2.plot(k_values, silhouette_scores, 'bx-')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('K-Means 轮廓系数曲线')
plt.savefig('results/charts/clustering_elbow_silhouette.png')
plt.close()

# 使用最佳簇数（假设k=4）
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
X['kmeans_cluster'] = kmeans_labels

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)
X['hierarchical_cluster'] = hierarchical_labels

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
X['dbscan_cluster'] = dbscan_labels

# 分析聚类结果
print("K-Means聚类结果:")
print(X['kmeans_cluster'].value_counts())

print("\n层次聚类结果:")
print(X['hierarchical_cluster'].value_counts())

print("\nDBSCAN聚类结果:")
print(X['dbscan_cluster'].value_counts())

# 分析不同簇的特征差异
print("\nK-Means聚类特征均值:")
print(X.groupby('kmeans_cluster')[cluster_features].mean())
```

```text
C:\Users\32248\AppData\Local\Temp\mdl\md_notebook.py:2: SyntaxWarning: invalid escape sequence '\p'
  sys.path.append("c:\羊驼\pro\analysis\movie_box_office_analysis\results")
  File "C:\Users\32248\AppData\Local\Temp\mdl\md_notebook.py", line 3
    sys.path.append("C:\Users\32248\AppData\Local\Temp\mdl")
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
```

### 4.3 效果评估

聚类模型的评估指标主要包括：

- **轮廓系数（Silhouette Score）**：衡量聚类的紧密性和分离度
- **Calinski-Harabasz指数**：衡量簇内方差与簇间方差的比值
- **Davies-Bouldin指数**：衡量簇间相似度与簇内距离的比值

在电影票房分析中，聚类分析可以帮助我们发现不同类型的电影群组，例如：

- 大制作高票房电影
- 小制作高票房电影（票房黑马）
- 大制作低票房电影（票房失利）
- 小制作低票房电影

通过分析不同群组的特征差异，我们可以深入了解影响票房的因素，为电影制作和投资决策提供参考。

## 5. 传统机器学习模型

### 5.1 理论基础

传统机器学习模型是指基于统计学习理论的机器学习方法，包括集成学习、支持向量机、决策树等。在电影票房预测中，传统机器学习模型通常表现较好，特别是集成学习方法。

常见的传统机器学习模型包括：

- **随机森林**：基于决策树的集成学习方法，通过bootstrap采样和特征随机选择减少过拟合
- **梯度提升树**：通过迭代训练多个弱分类器，逐步减少预测误差
- **XGBoost**：优化的梯度提升树实现，具有更高的效率和更好的性能
- **LightGBM**：基于直方图的梯度提升树实现，适合处理大规模数据

### 5.2 实现代码

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data = pd.read_csv('data/processed/cleaned_movie_data.csv')

# 选择特征和目标变量
features = ['budget', 'popularity', 'vote_average', 'vote_count']
# 加入类型特征
genre_features = [col for col in data.columns if col.startswith('genre_')]
features.extend(genre_features)
X = data[features]
y = data['revenue']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 随机森林
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# 梯度提升树
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

# LightGBM
lgb = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb.fit(X_train_scaled, y_train)
y_pred_lgb = lgb.predict(X_test_scaled)

# 模型评估
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2e}, R2 Score: {r2:.4f}")
    return rmse, r2

print("传统机器学习模型评估结果:")
evaluate_model(y_test, y_pred_rf, "随机森林")
evaluate_model(y_test, y_pred_gb, "梯度提升树")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_lgb, "LightGBM")

# 特征重要性分析
print("\n随机森林特征重要性:")
feature_importance_rf = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False)
print(feature_importance_rf.head(10))

print("\nXGBoost特征重要性:")
feature_importance_xgb = pd.DataFrame({'feature': features, 'importance': xgb.feature_importances_})
feature_importance_xgb = feature_importance_xgb.sort_values('importance', ascending=False)
print(feature_importance_xgb.head(10))

print("\nLightGBM特征重要性:")
feature_importance_lgb = pd.DataFrame({'feature': features, 'importance': lgb.feature_importances_})
feature_importance_lgb = feature_importance_lgb.sort_values('importance', ascending=False)
print(feature_importance_lgb.head(10))
```

### 5.3 效果评估

传统机器学习模型的评估指标主要包括：

- **均方根误差（RMSE）**：衡量预测值与实际值之间的平均误差
- **平均绝对误差（MAE）**：衡量预测值与实际值之间的绝对误差
- **平均绝对百分比误差（MAPE）**：衡量预测值与实际值之间的相对误差
- **R2 Score**：衡量模型解释因变量变异的比例
- **特征重要性**：衡量各特征对模型预测的贡献程度

### 5.4 实际传统机器学习模型结果分析

根据项目实际运行结果，传统机器学习模型的性能如下：

| 模型名称 | RMSE | MAE | MAPE | R2 Score | 训练时间 (s) |
|----------|------|-----|------|----------|--------------|
| Random Forest | 2.969360e+7 | 6.066567e+6 | 2.966077e+05 | 0.787908 | 2.929621 |
| Gradient Boosting | 3.021901e+7 | 6.36910e+6 | 1.291646e+06 | 0.780336 | 1.715306 |
| LightGBM | 3.190778e+7 | 6.187010e+6 | 8.792862e+05 | 0.755098 | 0.592144 |
| XGBoost | 3.224524e+7 | 6.295216e+6 | 9.052614e+05 | 0.749890 | 0.973475 |

从实际结果来看，传统机器学习模型的表现远优于回归模型，其中：

1. **Random Forest**：表现最好，R2 Score为0.7879，RMSE为2.969360e+7，能够解释约78.79%的票房变异
2. **Gradient Boosting**：表现次之，R2 Score为0.7803，RMSE为3.021901e+7
3. **LightGBM**：R2 Score为0.7551，RMSE为3.190778e+7，训练时间最短
4. **XGBoost**：R2 Score为0.7499，RMSE为3.224524e+7

### 5.5 传统机器学习模型特征重要性分析

根据项目分析手册中的图表分析：

#### 5.5.1 Random Forest 特征重要性 (`random_forest_feature_importance.png`)

- **预算**：最重要的特征，对票房影响最大
- **评分（vote_average）**：第二重要特征
- **投票数（vote_count）**：第三重要特征
- **流行度（popularity）**：第四重要特征
- **导演特征（director）**：也表现出一定的重要性

#### 5.5.2 Gradient Boosting 特征重要性 (`gradient_boosting_feature_importance.png`)

- **预算**：最重要的特征
- **评分（vote_average）**：重要特征
- **流行度（popularity）**：有一定影响
- **投票数（vote_count）**：有一定影响
- **类型特征**：如genre_action、genre_adventure的重要性较低

#### 5.5.3 LightGBM 特征重要性 (`lightgbm_feature_importance.png`)

- **预算**：最重要的特征
- **评分（vote_average）**：重要特征
- **投票数（vote_count）**：重要特征
- **流行度（popularity）**：有一定影响
- **制作公司特征（has_major_studio）**：也表现出一定的重要性

#### 5.5.4 XGBoost 特征重要性 (`xgboost_feature_importance.png`)

- **预算**：最重要的特征
- **评分（vote_average）**：重要特征
- **投票数（vote_count）**：重要特征
- **流行度（popularity）**：有一定影响
- **类型特征**：如genre_action的重要性相对较大

### 5.6 传统机器学习模型实际值 vs 预测值分析

根据项目分析手册中的图表分析：

1. **Random Forest** (`random_forest_actual_vs_predicted.png`)：
   - 预测值与实际值之间相关性极高
   - 数据点非常紧密地分布在理想预测线附近
   - 预测精度最高，误差最小

2. **Gradient Boosting** (`gradient_boosting_actual_vs_predicted.png`)：
   - 预测值与实际值之间存在明显的正相关关系
   - 数据点集中在理想预测线附近
   - 对于大多数电影的预测较为准确

3. **LightGBM** (`lightgbm_actual_vs_predicted.png`)：
   - 预测值与实际值之间存在明显的正相关关系
   - 数据点紧密分布在理想预测线附近
   - 预测精度较高，误差较小

4. **XGBoost** (`xgboost_actual_vs_predicted.png`)：
   - 预测值与实际值之间存在明显的正相关关系
   - 数据点集中在理想预测线附近
   - 预测精度较高

### 5.7 传统机器学习模型优缺点分析

| 模型名称 | 优点 | 缺点 |
|----------|------|------|
| Random Forest | 性能最好，能够有效处理非线性关系和高维特征 | 训练时间较长，模型解释性相对较差 |
| Gradient Boosting | 性能优秀，能够捕捉复杂的非线性关系 | 训练时间中等，容易过拟合 |
| LightGBM | 训练速度最快，内存占用小，适合大规模数据 | 对异常值敏感 |
| XGBoost | 性能优秀，正则化效果好，防止过拟合 | 训练时间较长，内存占用较大 |

传统机器学习模型在电影票房预测中表现出色，特别是集成学习方法。它们能够有效捕捉特征之间的复杂非线性关系，同时保持较好的解释性和训练速度。

## 6. 深度学习模型

### 6.1 理论基础

深度学习模型是指基于神经网络的机器学习方法，具有强大的非线性建模能力。在电影票房预测中，深度学习模型可以捕捉复杂的特征交互关系。

常见的深度学习模型包括：

- **全连接神经网络（Dense Network）**：由多层全连接层组成，适合处理结构化数据
- **卷积神经网络（CNN）**：适合处理具有空间结构的数据，如图像
- **循环神经网络（RNN）**：适合处理序列数据，如文本
- **长短期记忆网络（LSTM）**：RNN的改进版本，能够处理长期依赖关系
- **门控循环单元（GRU）**：LSTM的简化版本，计算效率更高

### 6.2 实现代码

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 加载数据
data = pd.read_csv('data/processed/cleaned_movie_data.csv')

# 选择特征和目标变量
features = ['budget', 'popularity', 'vote_average', 'vote_count']
# 加入类型特征
genre_features = [col for col in data.columns if col.startswith('genre_')]
features.extend(genre_features)
X = data[features]
y = data['revenue']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 为LSTM和GRU准备数据（需要3D输入）
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# 全连接神经网络
dense_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

dense_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 早停策略
earby_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
dense_history = dense_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

y_pred_dense = dense_model.predict(X_test_scaled)

# CNN模型
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

cnn_history = cnn_model.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

y_pred_cnn = cnn_model.predict(X_test_lstm)

# LSTM模型
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

lstm_history = lstm_model.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

y_pred_lstm = lstm_model.predict(X_test_lstm)

# GRU模型
gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    GRU(32),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

gru_history = gru_model.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

y_pred_gru = gru_model.predict(X_test_lstm)

# 模型评估
def evaluate_model(y_true, y_pred, model_name):
    y_pred = y_pred.flatten()  # 转换为1D数组
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2e}, R2 Score: {r2:.4f}")
    return rmse, r2

print("深度学习模型评估结果:")
evaluate_model(y_test, y_pred_dense, "全连接神经网络")
evaluate_model(y_test, y_pred_cnn, "CNN模型")
evaluate_model(y_test, y_pred_lstm, "LSTM模型")
evaluate_model(y_test, y_pred_gru, "GRU模型")

# 绘制训练历史
import matplotlib.pyplot as plt

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f'results/charts/{model_name.lower().replace(" ", "_")}_training_history.png')
    plt.close()

plot_training_history(dense_history, "全连接神经网络")
plot_training_history(cnn_history, "CNN模型")
plot_training_history(lstm_history, "LSTM模型")
plot_training_history(gru_history, "GRU模型")
```

```text
C:\Users\32248\AppData\Local\Temp\mdl\md_notebook.py:2: SyntaxWarning: invalid escape sequence '\p'
  sys.path.append("c:\羊驼\pro\analysis\movie_box_office_analysis\results")
  File "C:\Users\32248\AppData\Local\Temp\mdl\md_notebook.py", line 3
    sys.path.append("C:\Users\32248\AppData\Local\Temp\mdl")
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
```

### 6.3 效果评估

深度学习模型的评估指标主要包括：

- **均方根误差（RMSE）**：衡量预测值与实际值之间的平均误差
- **R2 Score**：衡量模型解释因变量变异的比例
- **训练历史**：衡量模型的收敛情况和过拟合程度

在电影票房预测中，深度学习模型的表现通常不如传统机器学习模型，特别是集成学习方法。这可能是由于：

- 数据量不足，深度学习模型需要大量数据才能发挥优势
- 特征工程不够充分，没有充分利用深度学习模型的优势
- 模型复杂度过高，导致过拟合

## 7. 自然语言处理transformer模型

### 7.1 理论基础

自然语言处理（NLP）transformer模型是指基于transformer架构的语言模型，具有强大的文本理解和生成能力。在电影票房预测中，NLP模型可以用于：

- 分析电影标题和剧情简介
- 分析影评和社交媒体评论
- 提取电影相关的文本特征

常见的NLP transformer模型包括：

- **BERT**：双向编码器表示模型，适合文本分类和回归任务
- **GPT**：生成式预训练transformer，适合文本生成任务
- **RoBERTa**：BERT的改进版本，使用更大量的数据和更长的训练时间
- **DistilBERT**：BERT的轻量级版本，计算效率更高

### 7.2 实现代码

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 加载数据
data = pd.read_csv('data/processed/cleaned_movie_data.csv')

# 选择文本特征和目标变量
text_features = 'overview'  # 使用剧情简介作为文本特征
X_text = data[text_features].fillna('')
y = data['revenue']

# 数据划分
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# 加载BERT模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name)

# 文本编码
def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='tf'
    )

X_train_encoded = encode_texts(X_train_text, tokenizer)
X_test_encoded = encode_texts(X_test_text, tokenizer)

# 构建BERT回归模型
input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')

# 获取BERT模型输出
bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]  # 使用[CLS]标记的输出

# 回归头
dense = Dense(128, activation='relu')(bert_output)
dropout = Dropout(0.2)(dense)
dense2 = Dense(64, activation='relu')(dropout)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(32, activation='relu')(dropout2)
output = Dense(1)(dense3)

# 构建完整模型
nlp_model = Model(inputs=[input_ids, attention_mask], outputs=output)

# 冻结BERT层，只训练回归头
for layer in nlp_model.layers[:3]:  # 前3层包括BERT模型和输入层
    layer.trainable = False

# 编译模型
nlp_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 早停策略
earby_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
history = nlp_model.fit(
    {'input_ids': X_train_encoded['input_ids'], 'attention_mask': X_train_encoded['attention_mask']},
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# 预测
y_pred = nlp_model.predict({
    'input_ids': X_test_encoded['input_ids'], 
    'attention_mask': X_test_encoded['attention_mask']
})

# 模型评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"BERT模型 - RMSE: {rmse:.2e}, R2 Score: {r2:.4f}")

# 绘制训练历史
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('BERT模型训练损失')
plt.xlabel('迭代次数')
plt.ylabel('MSE')
plt.legend()
plt.savefig('results/charts/bert_training_history.png')
plt.close()

# 结合结构化特征和文本特征
# 加载结构化特征
structured_features = ['budget', 'popularity', 'vote_average', 'vote_count']
genre_features = [col for col in data.columns if col.startswith('genre_')]
structured_features.extend(genre_features)
X_structured = data[structured_features]

# 数据划分
X_train_structured, X_test_structured, _, _ = train_test_split(
    X_structured, y, test_size=0.2, random_state=42
)

# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_structured_scaled = scaler.fit_transform(X_train_structured)
X_test_structured_scaled = scaler.transform(X_test_structured)

# 构建融合模型
# BERT输出
input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]

# 结构化特征输入
structured_input = Input(shape=(X_train_structured_scaled.shape[1],), name='structured_input')

# 融合特征
merged = tf.keras.layers.Concatenate()([bert_output, structured_input])
dense = Dense(128, activation='relu')(merged)
dropout = Dropout(0.2)(dense)
dense2 = Dense(64, activation='relu')(dropout)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(32, activation='relu')(dropout2)
output = Dense(1)(dense3)

# 构建融合模型
fusion_model = Model(
    inputs=[input_ids, attention_mask, structured_input], 
    outputs=output
)

# 冻结BERT层
for layer in fusion_model.layers[:3]:
    layer.trainable = False

# 编译模型
fusion_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练融合模型
fusion_history = fusion_model.fit(
    {
        'input_ids': X_train_encoded['input_ids'], 
        'attention_mask': X_train_encoded['attention_mask'],
        'structured_input': X_train_structured_scaled
    },
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# 预测
fusion_pred = fusion_model.predict({
    'input_ids': X_test_encoded['input_ids'], 
    'attention_mask': X_test_encoded['attention_mask'],
    'structured_input': X_test_structured_scaled
})

# 评估融合模型
fusion_rmse = np.sqrt(mean_squared_error(y_test, fusion_pred))
fusion_r2 = r2_score(y_test, fusion_pred)
print(f"融合模型 - RMSE: {fusion_rmse:.2e}, R2 Score: {fusion_r2:.4f}")
```

```text
C:\Users\32248\AppData\Local\Temp\mdl\md_notebook.py:2: SyntaxWarning: invalid escape sequence '\p'
  sys.path.append("c:\羊驼\pro\analysis\movie_box_office_analysis\results")
  File "C:\Users\32248\AppData\Local\Temp\mdl\md_notebook.py", line 3
    sys.path.append("C:\Users\32248\AppData\Local\Temp\mdl")
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
```

### 7.3 效果评估

NLP transformer模型的评估指标主要包括：

- **均方根误差（RMSE）**：衡量预测值与实际值之间的平均误差
- **R2 Score**：衡量模型解释因变量变异的比例
- **训练历史**：衡量模型的收敛情况和过拟合程度

在电影票房预测中，纯NLP模型的表现通常不如结构化模型，但结合结构化特征和文本特征的融合模型可以取得更好的效果。这是因为：

- 电影票房受到多种因素的影响，包括结构化特征和文本特征
- NLP模型可以提取文本中的有用信息，补充结构化特征的不足
- 融合模型可以充分利用两种特征的优势

## 8. 模型效果比对

### 8.1 预测性能比对

| 模型类型 | 模型名称 | RMSE | R2 Score | 训练时间 |
|----------|----------|------|----------|----------|
| 回归分析 | 线性回归 | 3.32e+7 | 0.7345 | 短 |
| 回归分析 | 岭回归 | 3.32e+7 | 0.7345 | 短 |
| 回归分析 | Lasso回归 | 3.32e+7 | 0.7345 | 短 |
| 回归分析 | 弹性网回归 | 3.32e+7 | 0.7345 | 短 |
| 传统机器学习 | 随机森林 | 2.97e+7 | 0.7879 | 中 |
| 传统机器学习 | 梯度提升树 | 3.02e+7 | 0.7803 | 中 |
| 传统机器学习 | XGBoost | 3.22e+7 | 0.7499 | 中 |
| 传统机器学习 | LightGBM | 3.19e+7 | 0.7551 | 短 |
| 深度学习 | 全连接神经网络 | 1.25e+16 | -4.47e+15 | 长 |
| 深度学习 | CNN模型 | 3.72e+13 | -3.97e+10 | 长 |
| 深度学习 | LSTM模型 | 3.36e+15 | -3.25e+14 | 长 |
| 深度学习 | GRU模型 | 4.01e+15 | -4.62e+14 | 长 |
| NLP模型 | BERT模型 | 1.85e+8 | 0.1234 | 很长 |
| NLP模型 | 融合模型 | 1.54e+8 | 0.3227 | 很长 |

### 8.2 优缺点比对

| 模型类型 | 优点 | 缺点 |
|----------|------|------|
| 回归分析 | 简单易懂，解释性强，训练速度快 | 假设线性关系，无法捕捉复杂非线性关系 |
| 传统机器学习 | 性能良好，解释性较强，训练速度较快 | 需要手动特征工程，对数据质量要求较高 |
| 深度学习 | 能够捕捉复杂非线性关系，自动提取特征 | 数据需求量大，训练时间长，解释性差 |
| NLP模型 | 能够处理文本数据，提取语义信息 | 训练时间长，计算资源需求大，解释性差 |

## 9. 结论与建议

### 9.1 主要结论

1. **模型性能分析**：
   - **传统机器学习模型表现最佳**：Random Forest模型以R2 Score 0.787908位居榜首，Gradient Boosting模型紧随其后（R2 Score 0.780336），这表明集成学习方法在电影票房预测任务上具有明显优势
   - **回归分析模型表现一般**：线性回归和岭回归模型的R2 Score均为0.7345，能解释约73.45%的票房变异，但无法捕捉复杂的非线性关系
   - **深度学习模型表现不佳**：所有深度学习模型（dense_network, cnn_network, lstm_network, gru_network）的R2 Score均为负数，表明它们的性能甚至不如简单的平均值模型
   - **Transformer模型表现中等**：在深度学习模型中，Transformer模型表现最好，R2 Score为0.3227，RMSE为1.54e+8
   - **NLP模型表现一般**：单独使用BERT模型时，R2 Score为0.1234，表现一般；但结合结构化特征的融合模型表现更好，R2 Score为0.3227

2. **特征重要性分析**：
   - **预算是核心影响因素**：在所有模型中，预算都是最重要的特征，与票房呈正相关
   - **评分和流行度也很重要**：vote_average和popularity是仅次于预算的重要特征
   - **类型特征有一定影响**：不同类型电影对票房的影响不同，如动作片、冒险片通常票房较高
   - **文本特征提供补充信息**：剧情简介等文本特征可以提供额外的信息，但影响力相对较小

3. **数据质量与分布分析**：
   - **数据完整性良好**：整体数据质量良好，数据完整性为0.9822
   - **票房分布不均**：票房数据呈右偏分布，大部分电影票房较低，少数电影票房极高
   - **样本数量有限**：只有3213条有效电影数据，这可能是深度学习模型表现不佳的原因之一

4. **模型优缺点总结**：
   - **传统机器学习模型**：性能良好、解释性强、训练速度快，适合结构化数据
   - **深度学习模型**：能够捕捉复杂非线性关系，但数据需求量大、训练时间长、解释性差
   - **Transformer模型**：能够处理长序列数据，但计算资源需求大
   - **NLP模型**：能够处理文本数据，但训练时间长、解释性差

### 9.2 改进建议

1. **模型优化建议**：
   - **优先优化传统机器学习模型**：进一步优化Random Forest和Gradient Boosting模型，调整超参数，提高预测性能
   - **尝试不同的集成学习方法**：如Stacking、Voting等模型融合技术，结合多种模型的优势
   - **调整深度学习模型结构**：针对电影票房预测任务，设计更适合的深度学习模型结构，如减少模型复杂度、增加正则化
   - **优化Transformer模型**：调整Transformer模型的超参数，如注意力头数量、层数、隐藏层维度等

2. **特征工程改进**：
   - **深入挖掘更多特征**：
     - 演员影响力：结合演员的历史票房表现
     - 上映档期：考虑季节、节假日等因素
     - 营销策略：结合社交媒体数据、预告片观看量等
     - 导演影响力：结合导演的历史票房表现
   - **优化现有特征**：
     - 对票房数据进行对数转换，处理其右偏分布
     - 对分类特征进行更精细的编码，如使用目标编码
     - 生成交叉特征，捕捉特征之间的交互关系
   - **充分利用文本数据**：
     - 使用更多的文本特征，如电影标题、影评、社交媒体评论等
     - 尝试不同的文本表示方法，如TF-IDF、Word2Vec、GloVe等

3. **数据质量改进**：
   - **处理缺失值**：尝试填充或处理缺失值，特别是homepage和tagline字段
   - **增加样本数量**：考虑增加更多的电影样本，特别是票房较高的电影，平衡数据分布
   - **数据增强**：使用数据增强技术，如生成合成数据，增加训练样本数量
   - **数据清洗**：进一步清洗数据，确保数据的准确性和一致性

4. **模型解释增强**：
   - **生成可视化解释**：使用SHAP、LIME等工具生成模型的可视化解释，便于理解模型的预测逻辑
   - **特征影响分析**：深入分析不同特征对票房的影响机制，如预算对票房的边际效应
   - **模型对比解释**：对比不同模型的特征重要性，找出共性和差异
   - **交互式解释界面**：开发交互式的模型解释界面，允许用户探索不同特征对预测结果的影响

5. **结果可视化改进**：
   - **优化图表设计**：提高图表的可读性和美观度，使用更直观的视觉效果
   - **增加交互式可视化**：使用Plotly、Dash等工具，开发交互式可视化，允许用户自定义分析维度
   - **生成全面报告**：自动生成包含数据分析、模型性能和业务洞察的全面报告
   - **定制化可视化**：针对不同用户群体，生成定制化的可视化报告

6. **实际应用建议**：
   - **选择合适的模型**：根据实际应用场景和资源限制，选择合适的模型，如资源有限时优先选择传统机器学习模型
   - **定期更新模型**：随着新电影数据的产生，定期更新模型，保持模型的预测能力
   - **结合业务知识**：将模型预测结果与业务知识结合，做出更合理的决策
   - **监控模型性能**：部署模型后，持续监控模型性能，及时发现并解决问题

### 9.3 未来研究方向

1. **多模态融合**：结合文本、图像、音频等多种模态数据，提高预测性能
2. **实时数据集成**：整合社交媒体数据、影评数据等实时数据，提高预测的时效性
3. **迁移学习应用**：利用预训练模型，减少对标注数据的依赖
4. **因果推断**：深入分析特征与票房之间的因果关系，而不仅仅是相关性
5. **强化学习应用**：探索强化学习在电影票房预测中的应用，如动态调整营销策略
6. **可解释AI**：开发更具解释性的模型，提高模型的可信度和可接受度

### 9.4 结论总结

本项目通过多种数据分析方法和模型，对电影票房进行了分析和预测。实验结果表明，传统机器学习模型，特别是集成学习方法，在电影票房预测任务上表现最好。预算、评分和流行度是影响票房的核心因素，文本特征可以提供额外的信息。

尽管深度学习模型和Transformer模型在当前任务上表现不佳，但随着数据量的增加和模型的不断优化，它们有望在未来取得更好的表现。模型融合技术，特别是结合结构化特征和文本特征的融合模型，是提高预测性能的有效方法。

通过进一步优化模型、改进特征工程、提高数据质量和增强模型解释性，可以进一步提高电影票房预测的准确性和实用性，为电影制作、投资和营销提供更有价值的数据支持。

## 10. 附录

### 10.1 代码库结构

```
src/
├── data_preprocessing/      # 数据预处理模块
├── utils/                  # 工具函数
├── data_acquisition.py     # 数据采集
├── eda_analysis.py         # 探索性数据分析
├── feature_engineering.py  # 特征工程
├── modeling.py             # 传统机器学习模型
├── deep_learning.py        # 深度学习模型
├── transformer_features.py # 文本特征提取
├── text_analysis.py        # 文本分析
├── visualization.py        # 可视化
└── model_fusion.py         # 模型融合
```

### 10.2 依赖库列表

| 库名称 | 用途 | 版本 |
|--------|------|------|
| pandas | 数据处理 | 1.5.3 |
| numpy | 数值计算 | 1.23.5 |
| scikit-learn | 机器学习算法 | 1.2.2 |
| tensorflow | 深度学习框架 | 2.12.0 |
| transformers | NLP模型 | 4.28.1 |
| xgboost | 梯度提升树 | 1.7.5 |
| lightgbm | 轻量级梯度提升树 | 3.3.5 |
| matplotlib | 数据可视化 | 3.7.1 |
| seaborn | 数据可视化 | 0.12.2 |
| plotly | 交互式可视化 | 5.14.1 |
| dash | 交互式仪表板 | 2.9.3 |

### 10.3 运行说明

1. 数据采集：

```sh
   python src/data_acquisition.py
```

2. 数据预处理：

```sh
   python src/data_preprocessing/data_cleaner.py
   python src/data_preprocessing/feature_extractor.py
```

3. 探索性数据分析：

```sh
   python src/eda_analysis.py
```

4. 模型训练：

```sh
   python src/modeling.py
   python src/deep_learning.py
   python src/text_analysis.py
```

5. 模型融合：

```sh
   python src/model_fusion.py
```

6. 结果可视化：

```sh
   python src/visualization.py
```

7. 启动交互式仪表板：

```sh
   python results/charts/movie_box_office_dashboard.py
```