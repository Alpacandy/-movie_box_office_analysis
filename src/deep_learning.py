import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Embedding, Layer
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import joblib
import warnings

# 获取当前脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, ".."))
# 将项目根目录添加到系统路径
sys.path.append(project_root)

# 导入项目内部模块
from src.utils.logging_config import get_logger
from src.utils.config_manager import global_config

warnings.filterwarnings('ignore')
logger = get_logger('deep_learning')

# 实现Transformer层
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"嵌入维度 {embed_dim} 必须能被头数 {num_heads} 整除")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        """获取配置，用于模型保存和加载"""
        return {
            'embed_dim': self.att.embed_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate
        }

class AdvancedTransformerBlock(Layer):
    """高级Transformer块，使用更高级的架构和激活函数"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(AdvancedTransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        # 使用GELU激活函数，比ReLU更适合Transformer
        self.ffn = Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        """获取配置，用于模型保存和加载"""
        return {
            'embed_dim': self.att.embed_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'dropout_rate': self.dropout1.rate
        }

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class PositionalEncoding(Layer):
    """位置编码层，为输入特征添加位置信息"""
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        position = tf.range(0, max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.cast(tf.range(0, embed_dim, 2), dtype=tf.float32) * -(tf.math.log(10000.0) / embed_dim))

        # 计算正弦和余弦位置编码
        sin_enc = tf.sin(position * div_term)
        cos_enc = tf.cos(position * div_term)

        # 交替拼接正弦和余弦编码
        pos_encoding = tf.zeros((max_len, embed_dim))
        pos_encoding = tf.concat([sin_enc, cos_enc], axis=1)[:, :embed_dim]

        # 添加批量维度
        pos_encoding = pos_encoding[tf.newaxis, :, :]

        # 将位置编码保存为非训练参数
        self.pe = tf.Variable(initial_value=pos_encoding, trainable=False)

    def call(self, x):
        """将位置编码添加到输入特征中"""
        # 获取输入序列长度
        seq_len = tf.shape(x)[1]

        # 将位置编码添加到输入特征中
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        """获取配置，用于模型保存和加载"""
        return {
            'embed_dim': self.pe.shape[-1],
            'max_len': self.pe.shape[1]
        }

# 设置可视化风格
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class DeepLearningModeling:
    def __init__(self, base_dir=None, results_dir=None):
        """
        初始化深度学习建模类，增强错误处理

        Args:
            base_dir (str): 数据基础目录
            results_dir (str): 结果保存目录
        """
        self.logger = get_logger('deep_learning.DeepLearningModeling')

        # 使用项目根目录作为基础路径
        self.base_dir = os.path.join(project_root, "data") if base_dir is None else base_dir
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.results_dir = os.path.join(project_root, "results") if results_dir is None else results_dir
        self.models_dir = os.path.join(self.results_dir, "models")
        self.charts_dir = os.path.join(self.results_dir, "charts")

        try:
            # 确保目录存在并检查写入权限
            for dir_path in [self.models_dir, self.charts_dir]:
                os.makedirs(dir_path, exist_ok=True)
                if not os.access(dir_path, os.W_OK):
                    self.logger.warning(f"目录 {dir_path} 不可写")
        except Exception as e:
            self.logger.warning(f"创建目录结构失败: {e}")

    def load_data(self, filename="feature_engineered_data.csv"):
        """
        加载特征工程后的数据，增强错误处理

        Args:
            filename (str): 数据文件名

        Returns:
            pd.DataFrame: 加载的数据或None
        """
        if not filename:
            self.logger.error("错误: 文件名不能为空")
            return None

        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"Error: File not found: {file_path}")
            self.logger.error("Please run the feature engineering script first: python feature_engineering.py")
            self.logger.error("The feature engineering script depends on the output of the data preprocessing script")
            return None

        self.logger.info(f"Loading data: {filename}")
        try:
            # 优化数据加载参数
            read_params = {
                'low_memory': False,
                'infer_datetime_format': True,
                'parse_dates': True
            }

            data = pd.read_csv(file_path, **read_params)
            self.logger.info(f"Data shape: {data.shape}")
            return data
        except Exception as e:
            self.logger.error(f"Error: Failed to load data: {e}")
            self.logger.error("Please check if the file format is correct")
            return None

    def prepare_data(self, data, target='revenue', test_size=0.2, random_state=42, scale=True):
        """
        准备训练数据和测试数据，增强错误处理

        Args:
            data (pd.DataFrame): 输入数据
            target (str): 目标列名
            test_size (float): 测试集比例
            random_state (int): 随机种子
            scale (bool): 是否进行数据标准化

        Returns:
            tuple: (X_train, X_test, y_train, y_test, features, scaler_X, scaler_y) 或 None
        """
        # 参数验证
        if data is None or data.empty:
            self.logger.error("Error: Input data is empty")
            return None

        if not isinstance(target, str) or not target:
            self.logger.error("Error: Target column name must be a valid string")
            return None

        if target not in data.columns:
            self.logger.error(f"Error: Target column '{target}' does not exist in the data")
            self.logger.error(f"Available columns: {list(data.columns)}")
            return None

        if not 0 < test_size < 1:
            self.logger.error(f"Error: Test set proportion {test_size} must be in the range (0, 1)")
            return None

        self.logger.info("Preparing training and testing data...")

        try:
            # 选择特征和目标变量
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            features = [col for col in numeric_cols if col not in [target, 'profit', 'return_on_investment']]

            X = data[features].values
            y = data[target].values.reshape(-1, 1)

            # 检查数据质量
            if np.isnan(X).any():
                self.logger.warning("警告: 特征数据中存在空值，将进行填充")
                X = np.nan_to_num(X, nan=np.nanmean(X))

            if np.isnan(y).any():
                self.logger.error("错误: 目标变量中存在空值")
                return None

            # 划分训练集和测试集
            # 检查release_year列是否存在
            stratify_param = None
            if 'release_year' in data.columns:
                stratify_param = data['release_year']//5
            elif 'release_month' in data.columns:
                stratify_param = data['release_month']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )

            # 数据缩放
            if scale:
                # 对特征进行缩放
                scaler_X = StandardScaler()
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)

                # 对目标变量进行缩放
                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train)
                y_test_scaled = scaler_y.transform(y_test)

                # 保存缩放器
                joblib.dump(scaler_X, os.path.join(self.models_dir, 'deep_learning_scaler_x.joblib'))
                joblib.dump(scaler_y, os.path.join(self.models_dir, 'deep_learning_scaler_y.joblib'))

                self.logger.info("Data scaling completed")
                return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, features, scaler_X, scaler_y

            return X_train, X_test, y_train, y_test, features, None, None
        except Exception as e:
            self.logger.error(f"错误: 数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_dense_network(self, input_shape, layers=None, dropout_rate=None, l2_reg=None):
        """构建更复杂的全连接神经网络，增加隐藏层和神经元数量，并使用更好的正则化"""
        # 从配置管理器获取默认参数
        if layers is None:
            layers = global_config.get('deep_learning.model_params.dense_network.layers', [256, 128, 64, 32])
        if dropout_rate is None:
            dropout_rate = global_config.get('deep_learning.model_params.dense_network.dropout_rate', 0.5)
        if l2_reg is None:
            l2_reg = global_config.get('deep_learning.model_params.dense_network.l2_reg', 0.001)

        model = Sequential()

        # 输入层
        model.add(Dense(layers[0], activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # 隐藏层
        for units in layers[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # 输出层
        model.add(Dense(1, activation='linear'))

        return model

    def build_cnn_network(self, input_shape, filters=[32, 64, 128], kernel_size=3, dropout_rate=0.3):
        """
        构建卷积神经网络，增强错误处理

        Args:
            input_shape (int): 输入特征数量
            filters (list): 卷积层过滤器数量列表
            kernel_size (int): 卷积核大小
            dropout_rate (float): Dropout率

        Returns:
            Sequential: 构建的CNN模型或None
        """
        # 参数验证
        if not isinstance(input_shape, int) or input_shape <= 0:
            self.logger.error(f"Error: Invalid input shape {input_shape}, must be a positive integer")
            return None

        if not filters or not all(isinstance(f, int) and f > 0 for f in filters):
            self.logger.error(f"Error: Invalid filter list {filters}")
            return None

        if kernel_size <= 0:
            self.logger.error(f"Error: Kernel size {kernel_size} must be greater than 0")
            return None

        if not 0 <= dropout_rate < 1:
            self.logger.error(f"Error: Dropout rate {dropout_rate} must be in the range [0, 1)")
            return None

        try:
            model = Sequential()

            # 调整输入形状以适应CNN
            model.add(Input(shape=(input_shape, 1)))

            # 卷积层
            for i, (filters_num, kernel) in enumerate(zip(filters, [kernel_size]*len(filters))):
                model.add(Conv1D(filters=filters_num, kernel_size=kernel, activation='relu', padding='same'))
                model.add(BatchNormalization())
                model.add(MaxPooling1D(pool_size=2, padding='same'))
                model.add(Dropout(dropout_rate))

            # 展平层
            model.add(Flatten())

            # 全连接层
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(dropout_rate))

            # 输出层
            model.add(Dense(1, activation='linear'))

            return model
        except Exception as e:
            self.logger.error(f"错误: 构建CNN网络失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_rnn_network(self, input_shape, units=[64, 32], dropout_rate=0.3, rnn_type='lstm'):
        """构建循环神经网络"""
        model = Sequential()

        # 调整输入形状以适应RNN
        model.add(Input(shape=(input_shape, 1)))

        # RNN层
        for i, units_num in enumerate(units):
            if rnn_type.lower() == 'lstm':
                if i == len(units) - 1:
                    model.add(LSTM(units_num, activation='relu', dropout=dropout_rate))
                else:
                    model.add(LSTM(units_num, activation='relu', dropout=dropout_rate, return_sequences=True))
            elif rnn_type.lower() == 'gru':
                if i == len(units) - 1:
                    model.add(GRU(units_num, activation='relu', dropout=dropout_rate))
                else:
                    model.add(GRU(units_num, activation='relu', dropout=dropout_rate, return_sequences=True))

        # 全连接层
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))

        # 输出层
        model.add(Dense(1, activation='linear'))

        return model

    def build_transformer_network(self, input_shape, embed_dim=128, num_heads=8, ff_dim=256, num_transformer_blocks=4, dropout_rate=0.3):
        """构建改进的Transformer网络，使用AdvancedTransformerBlock和GELU激活函数"""
        model = Sequential()

        # 直接使用原始输入形状，不需要额外维度
        model.add(Input(shape=(input_shape,)))

        # 将输入特征扩展到embed_dim，使用GELU激活函数
        model.add(Dense(embed_dim, activation='gelu'))

        # 将输入转换为3D张量，添加序列维度
        model.add(tf.keras.layers.Reshape((1, embed_dim)))

        # 添加位置编码
        model.add(PositionalEncoding(embed_dim, 1))

        # 添加多个高级Transformer块
        for _ in range(num_transformer_blocks):
            model.add(AdvancedTransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate))

        # 全局平均池化
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        # 全连接层，使用GELU激活函数
        model.add(Dense(128, activation='gelu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(64, activation='gelu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='gelu'))
        model.add(Dropout(dropout_rate))

        # 输出层
        model.add(Dense(1, activation='linear'))

        return model

    def compile_model(self, model, optimizer='adam', learning_rate=0.001):
        """编译模型"""
        # 选择优化器
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = Adam(learning_rate=learning_rate)

        # 编译模型
        model.compile(
            optimizer=opt,
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
        )

        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_name='model'):
        """
        训练模型，增强错误处理

        Args:
            model (keras.Model): 要训练的模型
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            epochs: 训练轮数
            batch_size: 批次大小
            model_name: 模型名称

        Returns:
            tuple: (训练好的模型, 训练历史)
        """
        self.logger.info(f"Training {model_name}...")

        # 定义回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f'{model_name}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # 训练模型
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # 保存完整训练历史
        np.save(os.path.join(self.models_dir, f'{model_name}_history.npy'), history.history)

        return model, history

    def evaluate_model(self, model, X_test, y_test, scaler_y=None):
        """评估模型"""
        # 预测
        y_pred = model.predict(X_test)

        # 如果使用了缩放器，反缩放
        if scaler_y is not None:
            y_test = scaler_y.inverse_transform(y_test)
            y_pred = scaler_y.inverse_transform(y_pred)

        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2 Score': r2
        }

        return metrics, y_test, y_pred

    def plot_training_history(self, history, model_name):
        """绘制训练历史"""
        # 1. 损失曲线
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mean_absolute_error'], label='Training MAE')
        plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        plt.title(f'{model_name} - Training and Validation MAE', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, f'{model_name.lower().replace(" ", "_")}_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"{model_name} training history plot saved")

    def plot_actual_vs_predicted(self, y_test, y_pred, model_name):
        """绘制实际值与预测值的对比图"""
        plt.figure(figsize=(12, 12))
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Revenue', fontsize=14)
        plt.ylabel('Predicted Revenue', fontsize=14)
        plt.title(f'Actual vs Predicted Revenue ({model_name})', fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # 添加评估指标文本
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR2 Score: {r2:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'))

        plt.savefig(os.path.join(self.charts_dir, f'{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"{model_name} actual vs predicted plot saved")

    def save_model(self, model, model_name):
        """保存训练好的模型"""
        model_path = os.path.join(self.models_dir, f'{model_name.lower().replace(" ", "_")}_model.h5')
        model.save(model_path)
        self.logger.info(f"Deep learning model saved to: {model_path}")
        return model_path

    def run_dense_network(self, X_train, X_test, y_train, y_test, scaler_y=None, model_name='Dense Network'):
        """运行全连接神经网络"""
        print("\n" + "=" * 50)
        self.logger.info(f"Training {model_name}")
        print("=" * 50)

        # 构建模型
        input_shape = X_train.shape[1]
        model = self.build_dense_network(input_shape, layers=[128, 64, 32, 16], dropout_rate=0.3, l2_reg=0.001)

        # 编译模型
        model = self.compile_model(model, optimizer='adam', learning_rate=0.001)

        # 训练模型
        model, history = self.train_model(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, model_name=model_name)

        # 评估模型
        metrics, y_test_inv, y_pred_inv = self.evaluate_model(model, X_test, y_test, scaler_y)

        # 显示评估结果
        self.logger.info(f"\n{model_name} evaluation results:")
        for metric, value in metrics.items():
              self.logger.info(f"  {metric}: {value:.4f}")

        # 绘制训练历史
        self.plot_training_history(history, model_name)

        # 绘制实际值与预测值对比图
        self.plot_actual_vs_predicted(y_test_inv, y_pred_inv, model_name)

        # 保存模型
        self.save_model(model, model_name)

        return model, metrics

    def run_cnn_network(self, X_train, X_test, y_train, y_test, scaler_y=None, model_name='CNN Network'):
        """运行卷积神经网络"""
        print("\n" + "=" * 50)
        self.logger.info(f"训练{model_name}")
        print("=" * 50)

        # 调整输入形状以适应CNN
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 构建模型
        input_shape = X_train.shape[1]
        model = self.build_cnn_network(input_shape, filters=[64, 32, 16], kernel_size=3, dropout_rate=0.3)

        # 编译模型
        model = self.compile_model(model, optimizer='adam', learning_rate=0.001)

        # 训练模型
        model, history = self.train_model(model, X_train_cnn, y_train, X_test_cnn, y_test, epochs=200, batch_size=32, model_name=model_name)

        # 评估模型
        metrics, y_test_inv, y_pred_inv = self.evaluate_model(model, X_test_cnn, y_test, scaler_y)

        # 显示评估结果
        self.logger.info(f"\n{model_name} evaluation results:")
        for metric, value in metrics.items():
              self.logger.info(f"  {metric}: {value:.4f}")

        # 绘制训练历史
        self.plot_training_history(history, model_name)

        # 绘制实际值与预测值对比图
        self.plot_actual_vs_predicted(y_test_inv, y_pred_inv, model_name)

        # 保存模型
        self.save_model(model, model_name)

        return model, metrics

    def run_rnn_network(self, X_train, X_test, y_train, y_test, scaler_y=None, rnn_type='lstm', model_name='LSTM Network'):
        """运行循环神经网络"""
        print("\n" + "=" * 50)
        self.logger.info(f"训练{model_name}")
        print("=" * 50)

        # 调整输入形状以适应RNN
        X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 构建模型
        input_shape = X_train.shape[1]
        model = self.build_rnn_network(input_shape, units=[64, 32], dropout_rate=0.3, rnn_type=rnn_type)

        # 编译模型
        model = self.compile_model(model, optimizer='adam', learning_rate=0.001)

        # 训练模型
        model, history = self.train_model(model, X_train_rnn, y_train, X_test_rnn, y_test, epochs=200, batch_size=32, model_name=model_name)

        # 评估模型
        metrics, y_test_inv, y_pred_inv = self.evaluate_model(model, X_test_rnn, y_test, scaler_y)

        # 显示评估结果
        self.logger.info(f"\n{model_name}评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # 绘制训练历史
        self.plot_training_history(history, model_name)

        # 绘制实际值与预测值对比图
        self.plot_actual_vs_predicted(y_test_inv, y_pred_inv, model_name)

        # 保存模型
        self.save_model(model, model_name)

        return model, metrics

    def run_transformer_network(self, X_train, X_test, y_train, y_test, scaler_y=None, model_name='Transformer Network'):
        """运行Transformer网络"""
        print("\n" + "=" * 50)
        self.logger.info(f"训练{model_name}")
        print("=" * 50)

        # 直接使用原始输入，不需要额外维度

        # 构建模型
        input_shape = X_train.shape[1]
        model = self.build_transformer_network(
            input_shape, 
            embed_dim=128, 
            num_heads=8, 
            ff_dim=256, 
            num_transformer_blocks=4, 
            dropout_rate=0.3
        )

        # 编译模型
        model = self.compile_model(model, optimizer='adam', learning_rate=0.001)

        # 训练模型
        model, history = self.train_model(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, model_name=model_name)

        # 评估模型
        metrics, y_test_inv, y_pred_inv = self.evaluate_model(model, X_test, y_test, scaler_y)

        # 显示评估结果
        self.logger.info(f"\n{model_name}评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # 绘制训练历史
        self.plot_training_history(history, model_name)

        # 绘制实际值与预测值对比图
        self.plot_actual_vs_predicted(y_test_inv, y_pred_inv, model_name)

        # 保存模型
        self.save_model(model, model_name)

        return model, metrics

    def compare_deep_models(self, results):
        """比较不同深度学习模型的性能"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Deep Learning Model Performance Comparison")
        self.logger.info("=" * 50)

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('R2 Score', ascending=False)

        # 显示排序后的结果
        self.logger.info("Deep learning model performance comparison results:")
        self.logger.info(results_df.to_string(index=False))

        # 可视化比较结果
        plt.figure(figsize=(14, 8))
        sns.barplot(x='R2 Score', y='Model', data=results_df, palette='magma')
        plt.title('Deep Learning Model Performance Comparison (R2 Score)', fontsize=16)
        plt.xlabel('R2 Score', fontsize=14)
        plt.ylabel('Model', fontsize=14)
        plt.xlim(0, 1)
        plt.savefig(os.path.join(self.charts_dir, 'deep_model_comparison_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Deep learning model R2 Score comparison plot saved")

        return results_df

    def run_complete_deep_learning(self, filename="feature_engineered_data.csv", target='revenue'):
        """运行完整的深度学习建模流程"""
        self.logger.info("=" * 60)
        self.logger.info("Starting complete deep learning modeling process")
        self.logger.info("=" * 60)

        # 1. 加载数据
        data = self.load_data(filename)
        if data is None:
            return None

        # 2. 准备数据
        X_train, X_test, y_train, y_test, features, scaler_X, scaler_y = self.prepare_data(data, target)

        # 3. 训练不同类型的深度学习模型
        results = []

        # 3.1 全连接神经网络
        dense_model, dense_metrics = self.run_dense_network(X_train, X_test, y_train, y_test, scaler_y, model_name='Dense Network')
        results.append({'Model': 'Dense Network', **dense_metrics})

        # 3.2 卷积神经网络
        cnn_model, cnn_metrics = self.run_cnn_network(X_train, X_test, y_train, y_test, scaler_y, model_name='CNN Network')
        results.append({'Model': 'CNN Network', **cnn_metrics})

        # 3.3 LSTM网络
        lstm_model, lstm_metrics = self.run_rnn_network(X_train, X_test, y_train, y_test, scaler_y, rnn_type='lstm', model_name='LSTM Network')
        results.append({'Model': 'LSTM Network', **lstm_metrics})

        # 3.4 GRU网络
        gru_model, gru_metrics = self.run_rnn_network(X_train, X_test, y_train, y_test, scaler_y, rnn_type='gru', model_name='GRU Network')
        results.append({'Model': 'GRU Network', **gru_metrics})

        # 3.5 Transformer网络
        transformer_model, transformer_metrics = self.run_transformer_network(X_train, X_test, y_train, y_test, scaler_y, model_name='Transformer Network')
        results.append({'Model': 'Transformer Network', **transformer_metrics})

        # 4. 比较模型
        results_df = self.compare_deep_models(results)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Deep learning modeling process completed!")
        self.logger.info("=" * 60)

        return results_df

    def run_quick_transformer(self, filename="feature_engineered_data.csv", target='revenue', epochs=1):
        """快速运行Transformer模型，减少epoch数量以加快训练速度"""
        self.logger.info("=" * 60)
        self.logger.info("Starting quick Transformer model training")
        self.logger.info("=" * 60)

        # 1. 加载数据
        data = self.load_data(filename)
        if data is None:
            return None

        # 2. 准备数据
        X_train, X_test, y_train, y_test, features, scaler_X, scaler_y = self.prepare_data(data, target)

        # 3. 构建模型
        input_shape = X_train.shape[1]
        model = self.build_transformer_network(
            input_shape, 
            embed_dim=128, 
            num_heads=8, 
            ff_dim=256, 
            num_transformer_blocks=4, 
            dropout_rate=0.3
        )

        # 编译模型
        model = self.compile_model(model, optimizer='adam', learning_rate=0.001)

        # 训练模型，使用较少的epoch
        model, history = self.train_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=32, model_name='Transformer Network')

        # 评估模型
        metrics, y_test_inv, y_pred_inv = self.evaluate_model(model, X_test, y_test, scaler_y)

        # 显示评估结果
        self.logger.info("\nTransformer Network评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # 绘制训练历史
        self.plot_training_history(history, 'Transformer Network')

        # 绘制实际值与预测值对比图
        self.plot_actual_vs_predicted(y_test_inv, y_pred_inv, 'Transformer Network')

        # 保存模型
        self.save_model(model, 'Transformer Network')

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Quick Transformer model training completed!")
        self.logger.info("=" * 60)

        return metrics

def main():
    """主函数，执行完整的深度学习建模流程"""
    dl_modeling = DeepLearningModeling()
    # 快速运行Transformer模型
    dl_modeling.run_quick_transformer()

if __name__ == "__main__":
    main()
