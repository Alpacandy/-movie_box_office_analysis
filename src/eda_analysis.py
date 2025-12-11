import os
import pandas as pd
import numpy as np
# 设置matplotlib为非交互式后端，避免线程问题
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
import logging
warnings.filterwarnings('ignore')

# 初始化日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eda_analysis')

# 设置可视化风格
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class EDAnalysis:
    def __init__(self, base_dir="./data", results_dir="./results"):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "processed")
        self.results_dir = results_dir

        # 创建递增编号的结果文件夹
        self.current_result_dir = self._create_result_dir()
        self.charts_dir = os.path.join(self.current_result_dir, "charts")

        os.makedirs(self.charts_dir, exist_ok=True)
        logger.info(f"结果将保存到: {self.current_result_dir}")

    def _create_result_dir(self):
        """创建递增编号的结果文件夹"""
        # 获取results目录下的所有文件夹
        result_dirs = []
        if os.path.exists(self.results_dir):
            for item in os.listdir(self.results_dir):
                item_path = os.path.join(self.results_dir, item)
                if os.path.isdir(item_path) and item.startswith("result_"):
                    result_dirs.append(item)

        # 计算下一个编号
        if not result_dirs:
            next_num = 1
        else:
            max_num = 0
            for dir_name in result_dirs:
                try:
                    num = int(dir_name.split('_')[1])
                    if num > max_num:
                        max_num = num
                except (ValueError, IndexError):
                    continue
            next_num = max_num + 1

        # 创建新的结果文件夹
        new_result_dir = os.path.join(self.results_dir, f"result_{next_num}")
        os.makedirs(new_result_dir, exist_ok=True)

        # 创建指向最新结果的软链接（Windows兼容性处理）
        latest_link = os.path.join(self.results_dir, "latest")
        if os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            else:
                os.rmdir(latest_link)

        # 在Windows系统上不创建符号链接（需要管理员权限）
        if os.name != 'nt':
            os.symlink(new_result_dir, latest_link, target_is_directory=True)

        return new_result_dir

    def load_data(self, filename="cleaned_movie_data.csv", use_dask=True, sample_size=None):
        """加载处理后的数据，支持Dask和采样

        Args:
            filename: 数据文件名
            use_dask: 是否使用Dask加载大规模数据
            sample_size: 采样大小，如果指定则返回采样数据

        Returns:
            加载的数据，可能是Pandas或Dask DataFrame
        """
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            logger.error("请先运行数据预处理脚本: python main.py --component preprocess")
            return None

        logger.info(f"正在加载数据: {filename}")

        try:
            # 尝试导入Dask
            try:
                import dask.dataframe as dd
                DASK_AVAILABLE = True
            except ImportError:
                DASK_AVAILABLE = False
                use_dask = False

            if use_dask and DASK_AVAILABLE:
                # 使用Dask加载大规模数据
                logger.info("使用Dask加载数据")
                data = dd.read_csv(file_path, low_memory=False)

                # 采样处理
                if sample_size:
                    logger.info(f"对数据进行采样，采样大小: {sample_size}")
                    # 先计算总行数
                    total_rows = len(data)
                    if sample_size < total_rows:
                        # 计算采样比例
                        sample_ratio = sample_size / total_rows
                        # 采样
                        data = data.sample(frac=sample_ratio, random_state=42)
                        # 将采样数据转换为Pandas DataFrame
                        data = data.compute()
                        logger.info(f"采样后数据形状: {data.shape}")
                    else:
                        logger.info("采样大小大于等于数据总量，返回全部数据")
                        data = data.compute()
            else:
                # 使用Pandas加载数据
                logger.info("使用Pandas加载数据")
                data = pd.read_csv(file_path, low_memory=False)
                logger.info(f"数据形状: {data.shape}")

                # 采样处理
                if sample_size and sample_size < len(data):
                    logger.info(f"对数据进行采样，采样大小: {sample_size}")
                    data = data.sample(n=sample_size, random_state=42)
                    logger.info(f"采样后数据形状: {data.shape}")

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            logger.error("请检查文件格式是否正确")
            return None

        return data

    def basic_statistics(self, data):
        """基础统计分析，支持Pandas和Dask DataFrame"""
        logger.info("=" * 50)
        logger.info("基础统计分析")
        logger.info("=" * 50)

        # 检查数据类型
        is_dask = hasattr(data, 'compute')

        if is_dask:
            logger.info("\n数据基本信息:")
            logger.info("   数据类型: Dask DataFrame")
            logger.info(f"   列数: {len(data.columns)}")
            logger.info(f"   行数: {len(data)}")
            logger.info(f"   列名: {list(data.columns)}")
            
            # 将Dask数据转换为Pandas DataFrame进行统计分析
            logger.info("将Dask数据转换为Pandas DataFrame进行统计分析")
            data = data.compute()
            is_dask = False
        
        if not is_dask:
            # 1. 基本信息
            logger.info("\n数据基本信息:")
            logger.info(data.info())

            # 2. 数值型数据统计
            logger.info("\n数值型数据统计:")
            logger.info(data.describe())

            # 3. 分类数据统计
            logger.info("\n分类数据统计:")
            categorical_cols = ['main_genre', 'director', 'has_major_studio']
            for col in categorical_cols:
                if col in data.columns:
                    logger.info(f"\n{col} 前10名:")
                    logger.info(data[col].value_counts().head(10))

        return data

    def analyze_revenue_distribution(self, data):
        """分析票房分布"""
        logger.info("\n" + "=" * 50)
        logger.info("票房分布分析")
        logger.info("=" * 50)

        # 1. 票房分布直方图
        plt.figure(figsize=(16, 8))
        sns.histplot(data['revenue'], bins=60, kde=True, color='#3498db', edgecolor='white', linewidth=1.5)
        plt.title('Distribution of Movie Revenue (USD)', fontsize=18, pad=20, fontweight='bold')
        plt.xlabel('Revenue (USD)', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')  # 添加虚线网格线
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'revenue_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("票房分布直方图已保存")

        # 2. 票房分布箱线图
        plt.figure(figsize=(16, 8))
        sns.boxplot(x=data['revenue'], color='#2ecc71', fliersize=5, linewidth=2)  # 调整颜色和线宽
        plt.title('Box Plot of Movie Revenue (USD)', fontsize=18, pad=20, fontweight='bold')
        plt.xlabel('Revenue (USD)', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')  # 添加虚线网格线
        plt.xticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'revenue_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("票房分布箱线图已保存")

        # 3. 票房按数量级分布
        data['revenue_bucket'] = pd.cut(data['revenue'], bins=[0, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9], 
                                      labels=['<1M', '1M-10M', '10M-50M', '50M-100M', '100M-500M', '>500M'])
        plt.figure(figsize=(14, 7))
        order = ['<1M', '1M-10M', '10M-50M', '50M-100M', '100M-500M', '>500M']
        sns.countplot(x='revenue_bucket', data=data, order=order, 
                     color='lightcoral', width=0.7)
        plt.title('Revenue Distribution by Buckets', fontsize=16, pad=20)
        plt.xlabel('Revenue Range (USD)', fontsize=14, labelpad=10)
        plt.ylabel('Count', fontsize=14, labelpad=10)
        plt.xticks(rotation=45, ha='center', fontsize=12)  # 调整x轴标签旋转角度和字体大小
        plt.grid(axis='y', alpha=0.3)  # 添加y轴网格线，提高可读性
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'revenue_buckets.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("票房数量级分布图已保存")

    def analyze_budget_revenue_relationship(self, data):
        """分析预算与票房关系"""
        logger.info("\n" + "=" * 50)
        logger.info("预算与票房关系分析")
        logger.info("=" * 50)

        # 1. 预算与票房散点图
        plt.figure(figsize=(16, 8))
        sns.scatterplot(x='budget', y='revenue', data=data, alpha=0.6, color='#e74c3c', 
                       edgecolor='white', s=80)  # 添加白色边框和调整点大小
        plt.title('Budget vs Revenue Relationship', fontsize=18, pad=20, fontweight='bold')
        plt.xlabel('Budget (USD)', fontsize=14, fontweight='bold')
        plt.ylabel('Revenue (USD)', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')  # 添加虚线网格线
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'budget_vs_revenue.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("预算与票房散点图已保存")

        # 2. 预算与回报率关系
        plt.figure(figsize=(16, 8))
        # 移除回报率异常值（超过100）
        roi_data = data[data['return_on_investment'] < 100]
        sns.scatterplot(x='budget', y='return_on_investment', data=roi_data, alpha=0.6, color='#9b59b6', 
                       edgecolor='white', s=80)  # 添加白色边框和调整点大小
        plt.title('Budget vs Return on Investment', fontsize=18, pad=20, fontweight='bold')
        plt.xlabel('Budget (USD)', fontsize=14, fontweight='bold')
        plt.ylabel('Return on Investment', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')  # 添加虚线网格线
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'budget_vs_roi.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("预算与回报率散点图已保存")

        # 3. 按制作公司类型分析
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='has_major_studio', y='revenue', data=data, showfliers=False)
        plt.title('Revenue by Studio Type', fontsize=16)
        plt.xlabel('Has Major Studio', fontsize=14)
        plt.ylabel('Revenue (USD)', fontsize=14)
        plt.yscale('log')
        plt.xticks([0, 1], ['Independent', 'Major Studio'])
        plt.savefig(os.path.join(self.charts_dir, 'revenue_by_studio_type.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("制作公司类型与票房关系图已保存")

    def analyze_genre_impact(self, data):
        """分析电影类型影响"""
        print("\n" + "=" * 50)
        print("电影类型影响分析")
        print("=" * 50)

        # 1. 各类型平均票房
        genre_stats = data.groupby('main_genre').agg({
            'revenue': ['mean', 'count'],
            'return_on_investment': 'mean',
            'vote_average': 'mean'
        }).reset_index()
        genre_stats.columns = ['main_genre', 'avg_revenue', 'movie_count', 'avg_roi', 'avg_rating']
        genre_stats = genre_stats.sort_values('avg_revenue', ascending=False)

        plt.figure(figsize=(14, 10))
        sns.barplot(y='main_genre', x='avg_revenue', data=genre_stats, palette='viridis')
        plt.title('Average Revenue by Movie Genre', fontsize=16)
        plt.xlabel('Average Revenue (USD)', fontsize=14)
        plt.ylabel('Genre', fontsize=14)
        plt.savefig(os.path.join(self.charts_dir, 'genre_vs_revenue.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("各类型平均票房图已保存")

        # 2. 各类型电影数量
        plt.figure(figsize=(14, 10))
        genre_counts = data['main_genre'].value_counts().sort_values()
        sns.barplot(y=genre_counts.index, x=genre_counts.values, palette='plasma')
        plt.title('Number of Movies by Genre', fontsize=16)
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Genre', fontsize=14)
        plt.savefig(os.path.join(self.charts_dir, 'genre_counts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("各类型电影数量图已保存")

        # 3. 类型与评分关系
        plt.figure(figsize=(14, 10))
        # 按平均评分排序类型
        genre_order = data.groupby('main_genre')['vote_average'].mean().sort_values(ascending=False).index
        sns.boxplot(y='main_genre', x='vote_average', data=data, showfliers=False, 
                   palette='coolwarm', order=genre_order, width=0.7)
        plt.title('Rating Distribution by Movie Genre', fontsize=16, pad=20)
        plt.xlabel('Average Rating', fontsize=14, labelpad=10)
        plt.ylabel('Genre', fontsize=14, labelpad=10)
        plt.xlim(0, 10)  # 设置x轴范围
        plt.grid(axis='x', alpha=0.3)  # 添加x轴网格线，提高可读性
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'genre_vs_rating.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("类型与评分关系图已保存")

    def analyze_temporal_trends(self, data):
        """分析时间趋势"""
        print("\n" + "=" * 50)
        print("时间趋势分析")
        print("=" * 50)

        # 1. 年度票房趋势
        yearly_stats = data.groupby('release_year').agg({
            'revenue': ['mean', 'count'],
            'budget': 'mean',
            'return_on_investment': 'mean'
        }).reset_index()
        yearly_stats.columns = ['release_year', 'avg_revenue', 'movie_count', 'avg_budget', 'avg_roi']

        plt.figure(figsize=(16, 8))
        sns.lineplot(x='release_year', y='avg_revenue', data=yearly_stats, marker='o', linewidth=2, color='blue')
        plt.title('Yearly Average Revenue Trend', fontsize=16, pad=20)
        plt.xlabel('Release Year', fontsize=14, labelpad=10)
        plt.ylabel('Average Revenue (USD)', fontsize=14, labelpad=10)
        plt.grid(True, alpha=0.3)  # 添加网格线，提高可读性
        plt.xticks(rotation=45, ha='center', fontsize=12)  # 调整x轴标签旋转角度和字体大小
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'yearly_revenue_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("年度票房趋势图已保存")

        # 2. 年度预算趋势
        plt.figure(figsize=(16, 8))
        sns.lineplot(x='release_year', y='avg_budget', data=yearly_stats, marker='o', linewidth=2, color='red')
        plt.title('Yearly Average Budget Trend', fontsize=16, pad=20)
        plt.xlabel('Release Year', fontsize=14, labelpad=10)
        plt.ylabel('Average Budget (USD)', fontsize=14, labelpad=10)
        plt.grid(True, alpha=0.3)  # 添加网格线，提高可读性
        plt.xticks(rotation=45, ha='center', fontsize=12)  # 调整x轴标签旋转角度和字体大小
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'yearly_budget_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("年度预算趋势图已保存")

        # 3. 月度票房分布
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='release_month', y='revenue', data=data, showfliers=False, palette='rainbow')
        plt.title('Revenue Distribution by Release Month', fontsize=16, pad=20)
        plt.xlabel('Release Month', fontsize=14, labelpad=10)
        plt.ylabel('Revenue (USD)', fontsize=14, labelpad=10)
        plt.yscale('log')
        plt.grid(axis='y', alpha=0.3)  # 添加y轴网格线，提高可读性
        plt.xticks(fontsize=12)  # 调整x轴标签字体大小
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'monthly_revenue.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("月度票房分布图已保存")

    def analyze_director_impact(self, data):
        """分析导演影响力"""
        print("\n" + "=" * 50)
        print("导演影响力分析")
        print("=" * 50)

        # 1. 导演平均票房（至少导演过3部电影）
        director_stats = data.groupby('director').agg({
            'revenue': ['mean', 'count'],
            'return_on_investment': 'mean'
        }).reset_index()
        director_stats.columns = ['director', 'avg_revenue', 'movie_count', 'avg_roi']
        director_stats = director_stats[director_stats['movie_count'] >= 3]
        top_directors = director_stats.sort_values('avg_revenue', ascending=False).head(15)

        plt.figure(figsize=(16, 12))
        sns.barplot(y='director', x='avg_revenue', data=top_directors, palette='coolwarm', width=0.8)
        plt.title('Top 15 Directors by Average Revenue (Min 3 Movies)', fontsize=18, pad=20)
        plt.xlabel('Average Revenue (USD)', fontsize=14, labelpad=10)
        plt.ylabel('Director', fontsize=14, labelpad=10)
        plt.grid(axis='x', alpha=0.3)  # 添加x轴网格线，提高可读性
        plt.yticks(fontsize=11)  # 调整y轴标签字体大小
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'top_directors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("顶级导演平均票房图已保存")

    def analyze_rating_impact(self, data):
        """分析评分影响力"""
        print("\n" + "=" * 50)
        print("评分影响力分析")
        print("=" * 50)

        # 1. 评分与票房关系
        plt.figure(figsize=(16, 8))
        sns.scatterplot(x='vote_average', y='revenue', data=data, alpha=0.6, color='green', edgecolor='white', s=80)
        plt.title('Rating vs Revenue Relationship', fontsize=18, pad=20)
        plt.xlabel('Average Rating', fontsize=14, labelpad=10)
        plt.ylabel('Revenue (USD)', fontsize=14, labelpad=10)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)  # 添加网格线，提高可读性
        plt.xlim(0, 10)  # 设置x轴范围
        plt.xticks(fontsize=12)  # 调整x轴标签字体大小
        plt.yticks(fontsize=12)  # 调整y轴标签字体大小
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'rating_vs_revenue.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("评分与票房关系图已保存")

        # 2. 评分分布
        plt.figure(figsize=(16, 8))
        sns.histplot(data['vote_average'], bins=20, kde=True, color='orange', edgecolor='white', alpha=0.8)
        plt.title('Distribution of Movie Ratings', fontsize=18, pad=20)
        plt.xlabel('Average Rating', fontsize=14, labelpad=10)
        plt.ylabel('Count', fontsize=14, labelpad=10)
        plt.grid(axis='y', alpha=0.3)  # 添加y轴网格线，提高可读性
        plt.xlim(0, 10)  # 设置x轴范围
        plt.xticks(fontsize=12)  # 调整x轴标签字体大小
        plt.yticks(fontsize=12)  # 调整y轴标签字体大小
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(self.charts_dir, 'rating_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("评分分布图已保存")

    def correlation_analysis(self, data):
        """相关性分析"""
        print("\n" + "=" * 50)
        print("相关性分析")
        print("=" * 50)

        # 1. 选择数值型列进行相关性分析
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        # 排除id等无关列
        relevant_cols = [col for col in numeric_cols if col not in ['id', 'release_year', 'release_month', 'release_day']]

        # 2. 计算相关系数
        correlation_matrix = data[relevant_cols].corr()

        # 3. 绘制热力图 - 优化文字重叠问题
        plt.figure(figsize=(30, 25))  # 大幅增大热力图尺寸
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # 仅保留绝对值大于0.15的相关性，进一步减少文字重叠
        threshold = 0.15
        filtered_correlation = correlation_matrix.copy()
        filtered_correlation[(np.abs(filtered_correlation) < threshold) & (filtered_correlation != 1.0)] = np.nan

        # 绘制热力图
        sns.heatmap(filtered_correlation, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                    square=True, linewidths=1.5, cbar_kws={'shrink': 0.7, 'pad': 0.02},
                    annot_kws={'size': 8, 'weight': 'bold', 'color': 'black'},  # 减小字体大小
                    vmin=-1, vmax=1, 
                    linecolor='white', 
                    edgecolor='white', 
                    cbar=True, 
                    robust=True)

        plt.title('Correlation Heatmap of Movie Features', fontsize=30, pad=30, fontweight='bold')
        # 大幅旋转x轴标签并调整对齐方式
        plt.xticks(rotation=60, ha='right', fontsize=10, rotation_mode='anchor', fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.tight_layout(pad=5.0)  # 增加内边距
        plt.subplots_adjust(bottom=0.25, left=0.25)  # 调整底部和左侧边距

        plt.savefig(os.path.join(self.charts_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("相关性热力图已保存")

        # 4. 与票房相关性最高的10个特征
        revenue_corr = correlation_matrix['revenue'].sort_values(ascending=False)
        logger.info("\n与票房相关性最高的10个特征:")
        logger.info(revenue_corr.head(10))

        return revenue_corr

    def cluster_analysis(self, data):
        """执行聚类分析，生成肘部曲线图和kmeans簇特征均值对比图"""
        print("\n" + "=" * 50)
        print("聚类分析")
        print("=" * 50)
        
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
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
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'clustering_elbow_silhouette.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("聚类肘部曲线图已保存")
        
        # 使用最佳簇数（假设k=4）
        k = 4
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        X['kmeans_cluster'] = kmeans_labels
        
        # 分析不同簇的特征差异
        print("\nK-Means聚类特征均值:")
        cluster_means = X.groupby('kmeans_cluster')[cluster_features].mean()
        print(cluster_means)
        
        # 绘制kmeans簇特征均值对比图
        cluster_means.plot(kind='bar', figsize=(12, 8))
        plt.title('Feature Mean Comparison Across K-Means Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Mean Value (Standardized)')
        plt.xticks(rotation=0)
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'kmeans_cluster_features_mean.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("K-Means簇特征均值对比图已保存")

    def run_complete_eda(self, filename="cleaned_movie_data.csv", sample_size=None):
        """运行完整的EDA分析"""
        # 1. 加载数据
        data = self.load_data(filename, sample_size=sample_size)
        if data is None:
            return None
            
        # 2. 确保数据为Pandas DataFrame
        if hasattr(data, 'compute'):
            logger.info("将Dask DataFrame转换为Pandas DataFrame")
            data = data.compute()

        # 3. 基础统计分析
        self.basic_statistics(data)

        # 3. 票房分布分析
        self.analyze_revenue_distribution(data)

        # 4. 预算与票房关系分析
        self.analyze_budget_revenue_relationship(data)

        # 5. 电影类型分析
        self.analyze_genre_impact(data)

        # 6. 时间趋势分析
        self.analyze_temporal_trends(data)

        # 7. 导演影响力分析
        self.analyze_director_impact(data)

        # 8. 评分影响力分析
        self.analyze_rating_impact(data)

        # 9. 相关性分析
        revenue_corr = self.correlation_analysis(data)
        
        # 10. 聚类分析
        self.cluster_analysis(data)

        logger.info("\n" + "=" * 50)
        logger.info("EDA分析完成！")
        logger.info(f"图表已保存到: {self.charts_dir}")
        logger.info("=" * 50)

        return revenue_corr


def main():
    """主函数，执行EDA分析流程"""
    parser = argparse.ArgumentParser(description='电影票房数据分析 - EDA分析')
    parser.add_argument('--filename', type=str, default='cleaned_movie_data.csv',
                        help='处理后的数据文件名，默认为cleaned_movie_data.csv')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='数据采样大小，如果指定则返回采样数据')

    args = parser.parse_args()

    eda = EDAnalysis()
    eda.run_complete_eda(filename=args.filename, sample_size=args.sample_size)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

