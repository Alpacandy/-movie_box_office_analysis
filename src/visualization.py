# 文档字符串和注释可以在导入之前

# 导入标准库模块
import os
import sys
import warnings
# 设置sys.path来导入本地模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入第三方库模块
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from tqdm import tqdm

# 导入本地模块
from src.utils.logging_config import setup_logging, get_logger


# 设置matplotlib为非交互式后端，避免线程问题
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境

# 设置警告过滤
warnings.filterwarnings('ignore')

# 设置可视化风格
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
# 设置中文显示支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 初始化日志系统
setup_logging()

# 初始化日志记录器
logger = get_logger('visualization')


class MovieVisualization:
    def __init__(self, base_dir=None, results_dir=None):
        # 使用项目根目录作为基础路径
        self.base_dir = os.path.join(project_root, "data") if base_dir is None else base_dir
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.results_dir = os.path.join(project_root, "results") if results_dir is None else results_dir
        self.charts_dir = os.path.join(self.results_dir, "charts")
        self.logger = get_logger('visualization.MovieVisualization')
        os.makedirs(self.charts_dir, exist_ok=True)

    def load_data(self, filename="cleaned_movie_data.csv", chunk_size=None, sample_fraction=None, random_state=42):
        """加载数据，支持分块加载和采样"""
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"文件不存在: {file_path}")
            self.logger.error("请先运行数据预处理脚本: python data_preprocessing.py")
            self.logger.error("数据预处理脚本依赖于数据获取脚本的输出")
            return None

        try:
            if sample_fraction and 0 < sample_fraction < 1:
                # 数据采样
                self.logger.info(f"正在对数据进行采样 ({sample_fraction * 100:.1f}%)...")
                data = pd.read_csv(file_path)
                return data.sample(frac=sample_fraction, random_state=random_state)
            elif chunk_size:
                # 分块加载
                self.logger.info(f"正在分块加载数据 (块大小: {chunk_size})...")
                chunks = []
                for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size)):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            else:
                # 常规加载
                self.logger.info("正在加载数据...")
                return pd.read_csv(file_path)
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            self.logger.error("请检查文件格式是否正确")
            return None

    def basic_visualizations(self, data, max_points=10000):
        """基础可视化，支持数据采样"""
        self.logger.info("正在生成基础可视化图表...")

        # 数据采样以提高性能
        if len(data) > max_points:
            self.logger.info(f"数据量过大 ({len(data)} 行)，将采样 {max_points} 行用于可视化...")
            data_sample = data.sample(n=max_points, random_state=42)
        else:
            data_sample = data

        # 1. 票房分布直方图
        plt.figure(figsize=(14, 7))
        sns.histplot(data_sample['revenue'], bins=50, kde=True, color='skyblue')
        plt.title('电影票房分布', fontsize=16)
        plt.xlabel('票房 (美元)', fontsize=14)
        plt.ylabel('电影数量', fontsize=14)
        plt.xscale('log')
        plt.savefig(os.path.join(self.charts_dir, 'basic_revenue_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("票房分布直方图已保存")

        # 2. 预算与票房散点图
        plt.figure(figsize=(14, 7))
        sns.scatterplot(x='budget', y='revenue', data=data_sample, alpha=0.6, color='coral')
        plt.title('预算与票房关系', fontsize=16)
        plt.xlabel('预算 (美元)', fontsize=14)
        plt.ylabel('票房 (美元)', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(self.charts_dir, 'basic_budget_vs_revenue.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("预算与票房散点图已保存")

        # 3. 电影类型分布 - 兼容不同的数据结构
        plt.figure(figsize=(14, 10))

        if 'main_genre' in data.columns:
            # 使用原始的main_genre列
            genre_counts = data['main_genre'].value_counts().sort_values()
            sns.barplot(y=genre_counts.index, x=genre_counts.values, palette='viridis')
        else:
            # 使用one-hot编码的genre列
            genre_columns = [col for col in data.columns if col.startswith('genre_')]
            if genre_columns:
                # 计算每个类型的电影数量
                genre_counts = data[genre_columns].sum().sort_values()
                # 去掉genre_前缀
                genre_counts.index = genre_counts.index.str.replace('genre_', '', regex=False)
                sns.barplot(y=genre_counts.index, x=genre_counts.values, palette='viridis')
            else:
                self.logger.warning("未找到电影类型相关列，跳过电影类型分布图")
                plt.close()
                return True

        plt.title('电影类型分布', fontsize=16)
        plt.xlabel('电影数量', fontsize=14)
        plt.ylabel('类型', fontsize=14)
        plt.savefig(os.path.join(self.charts_dir, 'basic_genre_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("电影类型分布图已保存")

        return True

    def interactive_visualizations(self, data, max_points=10000):
        """生成交互式可视化，支持数据采样"""
        self.logger.info("正在生成交互式可视化图表...")

        # 数据采样以提高性能和减少HTML文件大小
        if len(data) > max_points:
            self.logger.info(f"数据量过大 ({len(data)} 行)，将采样 {max_points} 行用于交互式散点图...")
            scatter_data = data.sample(n=max_points, random_state=42)
        else:
            scatter_data = data

        # 确定可用的标题列
        title_column = 'title'
        if title_column not in scatter_data.columns:
            # 尝试使用title_x或title_y
            if 'title_x' in scatter_data.columns:
                title_column = 'title_x'
            elif 'title_y' in scatter_data.columns:
                title_column = 'title_y'

        # 1. 票房与预算的交互式散点图 - 兼容不同的数据结构
        if 'main_genre' in scatter_data.columns:
            # 使用原始的main_genre列
            fig = px.scatter(
                scatter_data, x='budget', y='revenue', color='main_genre',
                title='Budget vs Revenue by Movie Genre',
                labels={'budget': 'Budget (USD)', 'revenue': 'Revenue (USD)'},
                hover_name=title_column, opacity=0.7, log_x=True, log_y=True
            )
        else:
            # 尝试使用最主要的类型特征或使用其他特征作为颜色区分
            fig = px.scatter(
                scatter_data, x='budget', y='revenue',
                title='Budget vs Revenue Relationship',
                labels={'budget': 'Budget (USD)', 'revenue': 'Revenue (USD)'},
                hover_name=title_column, opacity=0.7, log_x=True, log_y=True
            )

        fig.update_layout(font=dict(size=12), width=1000, height=600, legend_title='Movie Genre')
        fig.write_html(os.path.join(self.charts_dir, 'interactive_budget_vs_revenue.html'))
        self.logger.info("交互式预算与票房散点图已保存")

        # 2. 年度票房趋势
        if 'release_year' in data.columns:
            yearly_stats = data.groupby('release_year').agg({
                'revenue': 'mean',
                'budget': 'mean',
                'id': 'count'
            }).reset_index()
            yearly_stats.columns = ['release_year', 'avg_revenue', 'avg_budget', 'movie_count']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=yearly_stats['release_year'], y=yearly_stats['avg_revenue'],
                           mode='lines+markers', name='Average Revenue', line=dict(color='blue')),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=yearly_stats['release_year'], y=yearly_stats['avg_budget'],
                           mode='lines+markers', name='Average Budget', line=dict(color='red')),
                secondary_y=True
            )
            fig.update_layout(
                title='Yearly Average Revenue and Budget Trend',
                xaxis_title='Release Year',
                yaxis_title='Average Revenue (USD)',
                yaxis2_title='Average Budget (USD)',
                width=1000, height=600, font=dict(size=12)
            )
            fig.write_html(os.path.join(self.charts_dir, 'interactive_yearly_trend.html'))
            self.logger.info("交互式年度趋势图已保存")
        else:
            self.logger.warning("未找到release_year列，跳过年度趋势图")

        # 3. 导演票房排行榜
        if 'director' in data.columns:
            director_stats = data.groupby('director').agg({
                'revenue': 'mean',
                'id': 'count'
            }).reset_index()
            director_stats.columns = ['director', 'avg_revenue', 'movie_count']
            top_directors = director_stats[
                director_stats['movie_count'] >= 3
            ].sort_values('avg_revenue', ascending=False).head(20)

            fig = px.bar(
                top_directors, y='director', x='avg_revenue',
                title='Top 20 Directors by Average Revenue '
                      '(Min 3 Movies)',
                labels={
                    'director': 'Director', 
                    'avg_revenue': 'Average Revenue (USD)'
                },
                color='avg_revenue',
                color_continuous_scale='viridis',
                hover_data=['movie_count']
            )
            fig.update_layout(font=dict(size=12), width=1000, height=800)
            fig.write_html(os.path.join(self.charts_dir, 'interactive_top_directors.html'))
            self.logger.info("交互式导演排行榜已保存")
        else:
            self.logger.warning("未找到director列，跳过导演排行榜")

        return True

    def create_dashboard(self, data):
        """创建交互式仪表板"""
        self.logger.info("正在创建交互式仪表板...")

        # 数据采样以优化仪表盘性能
        if len(data) > 50000:
            self.logger.info(f"数据量过大 ({len(data)} 行)，将采样 50000 行用于仪表盘...")
            dashboard_data = data.sample(n=50000, random_state=42)
        else:
            dashboard_data = data

        # 创建Dash应用
        app = dash.Dash(__name__, title='电影票房分析仪表板')

        # 布局设计
        app.layout = html.Div([
            html.H1("电影票房分析仪表板", style={'textAlign': 'center', 'marginBottom': '30px'}),

            # 第一行：筛选器
            html.Div([
                # 根据数据结构选择合适的筛选器
                'release_year' in dashboard_data.columns and html.Div([
                    html.Label("年份范围:"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=dashboard_data['release_year'].min(),
                        max=dashboard_data['release_year'].max(),
                        step=1,
                        value=[dashboard_data['release_year'].min(), dashboard_data['release_year'].max()],
                        marks={str(year): str(year) for year in range(
                            dashboard_data['release_year'].min(),
                            dashboard_data['release_year'].max() + 1, 5
                        )}
                    )
                ], style={'width': '65%', 'display': 'inline-block', 'marginTop': '20px'})
            ], style={'marginBottom': '30px'}),

            # 第二行：图表
            html.Div([
                html.Div([
                    dcc.Graph(id='revenue-distribution')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    dcc.Graph(id='budget-vs-revenue')
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '30px'}),

            # 第三行：图表
            html.Div([
                'release_year' in dashboard_data.columns and html.Div([
                    dcc.Graph(id='yearly-revenue')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

                (('main_genre' in dashboard_data.columns)
                 or [col for col in dashboard_data.columns if col.startswith('genre_')])
                and html.Div([
                    dcc.Graph(id='genre-revenue')
                ], style={'width': '48%', 'display': 'inline-block'})
            ])
        ])

        # 回调函数
        @app.callback(
            [Output('revenue-distribution', 'figure'),
             Output('budget-vs-revenue', 'figure'),
             Output('yearly-revenue', 'figure'),
             Output('genre-revenue', 'figure')],
            [Input('year-slider', 'value')]
        )
        def update_charts(year_range):
            # 筛选数据
            filtered_data = dashboard_data.copy()

            # 年份筛选
            if 'release_year' in filtered_data.columns:
                filtered_data = filtered_data[
                    (filtered_data['release_year'] >= year_range[0])
                    & (filtered_data['release_year'] <= year_range[1])
                ]

            # 1. 票房分布
            fig1 = px.histogram(
                filtered_data, x='revenue',
                title='Distribution of Movie Revenue',
                labels={'revenue': 'Revenue (USD)'},
                nbins=50, opacity=0.7, log_x=True
            )
            fig1.update_layout(width=500, height=400)

            # 2. 预算与票房
            fig2 = px.scatter(
                filtered_data, x='budget', y='revenue',
                title='Budget vs Revenue',
                labels={'budget': 'Budget (USD)', 'revenue': 'Revenue (USD)'},
                opacity=0.7, log_x=True, log_y=True
            )
            fig2.update_layout(width=500, height=400, showlegend=False)

            # 3. 年度票房趋势
            if 'release_year' in filtered_data.columns:
                yearly_stats = filtered_data.groupby('release_year')['revenue'].mean().reset_index()
                fig3 = px.line(
                    yearly_stats, x='release_year', y='revenue',
                    title='Yearly Average Revenue',
                    labels={
                        'release_year': 'Release Year',
                        'revenue': 'Average Revenue (USD)'
                    },
                    markers=True
                )
                fig3.update_layout(width=500, height=400)
            else:
                # 创建一个空图表
                fig3 = px.line(
                    title='Yearly Average Revenue',
                    labels={'release_year': 'Release Year', 'revenue': 'Average Revenue (USD)'}
                )
                fig3.update_layout(width=500, height=400)

            # 4. 类型票房 - 兼容不同的数据结构
            if 'main_genre' in filtered_data.columns:
                genre_stats = filtered_data.groupby('main_genre')['revenue'].mean().reset_index()
                genre_stats = genre_stats.sort_values('revenue', ascending=False)
                fig4 = px.bar(
                    genre_stats, y='main_genre', x='revenue',
                    title='Average Revenue by Genre',
                    labels={'main_genre': 'Genre', 'revenue': 'Average Revenue (USD)'},
                    color='revenue', color_continuous_scale='viridis'
                )
                fig4.update_layout(width=500, height=400)
            else:
                # 使用one-hot编码的genre列
                genre_columns = [col for col in filtered_data.columns if col.startswith('genre_')]
                if genre_columns:
                    # 计算每个类型的平均票房
                    genre_stats = filtered_data.groupby(
                        filtered_data[genre_columns].idxmax(axis=1)
                    )['revenue'].mean().reset_index()
                    genre_stats.columns = ['genre', 'revenue']
                    # 去掉genre_前缀
                    genre_stats['genre'] = genre_stats['genre'].str.replace('genre_', '', regex=False)
                    genre_stats = genre_stats.sort_values('revenue', ascending=False)

                    fig4 = px.bar(
                        genre_stats, y='genre', x='revenue',
                        title='Average Revenue by Genre',
                        labels={
                            'genre': 'Genre',
                            'revenue': 'Average Revenue (USD)'
                        },
                        color='revenue', color_continuous_scale='viridis'
                    )
                    fig4.update_layout(width=500, height=400)
                else:
                    # 创建一个空图表
                    fig4 = px.bar(
                        title='Average Revenue by Genre',
                        labels={'genre': 'Genre', 'revenue': 'Average Revenue (USD)'}
                    )
                    fig4.update_layout(width=500, height=400)

            return fig1, fig2, fig3, fig4

        # 保存仪表板代码
        dashboard_code = '''# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 加载数据（支持大规模数据采样）
def load_data_with_sampling(filename='data/processed/cleaned_movie_data.csv', max_rows=100000, random_state=42):
    """加载数据并进行采样以优化性能"""
    data = pd.read_csv(filename)
    if len(data) > max_rows:
        return data.sample(n=max_rows, random_state=random_state)
    return data

# 加载数据
data = load_data_with_sampling()

# 创建Dash应用
app = dash.Dash(__name__, title='电影票房分析仪表板', suppress_callback_exceptions=True)

# 定义颜色主题
def get_theme(theme):
    if theme == 'dark':
        return {
            'background': '#1a1a1a',
            'text': '#ffffff',
            'card': '#2d2d2d',
            'border': '#404040',
            'grid': '#333333'
        }
    else:
        return {
            'background': '#f5f5f5',
            'text': '#000000',
            'card': '#ffffff',
            'border': '#e0e0e0',
            'grid': '#e0e0e0'
        }

# 布局设计
app.layout = html.Div([
    # 主题切换
    html.Div([
        html.H1("电影票房分析仪表板", style={
            'textAlign': 'center',
            'marginBottom': '30px'
        }),
        html.Div([
            dcc.RadioItems(
                id='theme-toggle',
                options=[{'label': '浅色模式', 'value': 'light'}, {'label': '深色模式', 'value': 'dark'}],
                value='light',
                inline=True,
                style={'marginBottom': '20px'}
            )
        ], style={'textAlign': 'center'})
    ]),
    
    # 主内容容器
    html.Div(id='main-container', style={'padding': '20px', 'minHeight': '100vh'})
])

@app.callback(
    Output('main-container', 'children'),
    [Input('theme-toggle', 'value')]
)
def update_theme(theme='light'):
    theme_colors = get_theme(theme)

    return html.Div([
        # 第一行：筛选器
        html.Div([
            html.Div([
                html.Label("电影类型:", style={'color': theme_colors['text']}),
                dcc.Dropdown(
                    id='genre-filter',
                    options=[{'label': genre, 'value': genre} for genre in data['main_genre'].unique()],
                    value=None,
                    placeholder="选择电影类型",
                    multi=True,
                    style={'marginBottom': '10px', 'color': theme_colors['text']}
                )
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("导演:", style={'color': theme_colors['text']}),
                dcc.Dropdown(
                    id='director-filter',
                    options=[
                        {'label': director, 'value': director} 
                        for director in data['director'].value_counts().head(50).index
                    ],
                    value=None,
                    placeholder="选择导演",
                    multi=True,
                    style={'marginBottom': '10px', 'color': theme_colors['text']}
                )
            ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("年份范围:", style={'color': theme_colors['text']}),
                dcc.RangeSlider(
                    id='year-slider',
                    min=data['release_year'].min(),
                    max=data['release_year'].max(),
                    step=1,
                    value=[data['release_year'].min(), data['release_year'].max()],
                    marks={
                        str(year): str(year) 
                        for year in range(
                            data['release_year'].min(), 
                            data['release_year'].max() + 1, 
                            5
                        )
                    },
                    tooltip={'always_visible': True, 'placement': 'bottom'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginTop': '20px'})
        ], style={
            'marginBottom': '30px',
            'padding': '20px',
            'backgroundColor': theme_colors['card'],
            'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'border': f'1px solid {theme_colors["border"]}'
        }),

        # 第二行：概览卡片
        html.Div([
            html.Div([
                html.H3("总电影数", style={'color': theme_colors['text']}),
                html.P(len(data), style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#2563eb'})
            ], style={
                'width': '24%', 
                'display': 'inline-block', 
                'marginRight': '1%',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}',
                'textAlign': 'center'
            }),

            html.Div([
                html.H3("平均票房", style={'color': theme_colors['text']}),
                html.P(f"${data['revenue'].mean():,.2f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#10b981'})
            ], style={
                'width': '24%', 
                'display': 'inline-block', 
                'marginRight': '1%',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}',
                'textAlign': 'center'
            }),

            html.Div([
                html.H3("平均评分", style={'color': theme_colors['text']}),
                html.P(f"{data['vote_average'].mean():.2f}/10", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#f59e0b'})
            ], style={
                'width': '24%', 
                'display': 'inline-block', 
                'marginRight': '1%',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}',
                'textAlign': 'center'
            }),

            html.Div([
                html.H3("平均预算", style={'color': theme_colors['text']}),
                html.P(f"${data['budget'].mean():,.2f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#ef4444'})
            ], style={
                'width': '24%', 'display': 'inline-block',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}',
                'textAlign': 'center'
            })
        ], style={'marginBottom': '30px'}),

        # 第二行：图表
        html.Div([
            html.Div([
                html.H3("票房分布", style={'color': theme_colors['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='revenue-distribution', style={'height': '400px'})
            ], style={
                'width': '48%', 'display': 'inline-block', 'marginRight': '2%',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}'
            }),

            html.Div([
                html.H3("预算与票房关系", style={'color': theme_colors['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='budget-vs-revenue', style={'height': '400px'})
            ], style={
                'width': '48%', 'display': 'inline-block',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}'
            })
        ], style={'marginBottom': '30px'}),

        # 第三行：图表
        html.Div([
            html.Div([
                html.H3("年度票房趋势", style={'color': theme_colors['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='yearly-revenue', style={'height': '400px'})
            ], style={
                'width': '48%', 'display': 'inline-block', 'marginRight': '2%',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}'
            }),

            html.Div([
                html.H3("电影类型票房分析", style={'color': theme_colors['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='genre-revenue', style={'height': '400px'})
            ], style={
                'width': '48%', 'display': 'inline-block',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}'
            })
        ], style={'marginBottom': '30px'}),

        # 第四行：图表
        html.Div([
            html.Div([
                html.H3("评分与票房关系", style={'color': theme_colors['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='rating-vs-revenue', style={'height': '400px'})
            ], style={
                'width': '48%', 'display': 'inline-block', 'marginRight': '2%',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}'
            }),

            html.Div([
                html.H3("导演影响力", style={'color': theme_colors['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='director-impact', style={'height': '400px'})
            ], style={
                'width': '48%', 'display': 'inline-block',
                'padding': '20px',
                'backgroundColor': theme_colors['card'],
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'border': f'1px solid {theme_colors["border"]}'
            })
        ], style={'marginBottom': '30px'}),

        # 第五行：数据表格
        html.Div([
            html.H3("电影数据表格", style={'color': theme_colors['text'], 'marginBottom': '10px'}),
            dash_table.DataTable(
                id='movie-table',
                columns=[
                    {'name': '标题', 'id': 'title'},
                    {'name': '年份', 'id': 'release_year'},
                    {'name': '类型', 'id': 'main_genre'},
                    {'name': '导演', 'id': 'director'},
                    {'name': '预算', 'id': 'budget', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                    {'name': '票房', 'id': 'revenue', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                    {'name': '评分', 'id': 'vote_average', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                data=data.head(100).to_dict('records'),
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'backgroundColor': theme_colors['card'],
                    'color': theme_colors['text'],
                    'border': f'1px solid {theme_colors["border"]}'
                },
                style_header={
                    'backgroundColor': theme_colors['card'],
                    'fontWeight': 'bold',
                    'border': f'1px solid {theme_colors["border"]}'
                },
                page_size=10,
                filter_action='native',
                sort_action='native',
                sort_mode='multi'
            )
        ], style={
            'padding': '20px',
            'backgroundColor': theme_colors['card'],
            'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'border': f'1px solid {theme_colors["border"]}'
        })
    ], style={'backgroundColor': theme_colors['background']})

# 回调函数
@app.callback(
    [Output('revenue-distribution', 'figure'),
     Output('budget-vs-revenue', 'figure'),
     Output('yearly-revenue', 'figure'),
     Output('genre-revenue', 'figure'),
     Output('rating-vs-revenue', 'figure'),
     Output('director-impact', 'figure'),
     Output('movie-table', 'data')],
    [Input('genre-filter', 'value'),
     Input('director-filter', 'value'),
     Input('year-slider', 'value'),
     Input('theme-toggle', 'value')]
)
def update_charts(genre_filter, director_filter, year_range, theme):
    # 筛选数据
    filtered_data = data.copy()

    if genre_filter:
        filtered_data = filtered_data[filtered_data['main_genre'].isin(genre_filter)]

    if director_filter:
        filtered_data = filtered_data[filtered_data['director'].isin(director_filter)]

    filtered_data = filtered_data[
        (filtered_data['release_year'] >= year_range[0]) & 
        (filtered_data['release_year'] <= year_range[1])
    ]

    # 设置主题
    theme_colors = get_theme(theme)
    plot_bgcolor = theme_colors['card']
    paper_bgcolor = theme_colors['card']
    font_color = theme_colors['text']
    grid_color = theme_colors['grid']

    # 1. 票房分布
    fig1 = px.histogram(
        filtered_data, x='revenue',
        title='票房分布',
        labels={'revenue': '票房 (美元)', 'count': '电影数量'},
        color_discrete_sequence=['#2563eb']
    )
    fig1.update_layout(
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font_color=font_color,
        xaxis={'gridcolor': grid_color},
        yaxis={'gridcolor': grid_color}
    )

    # 2. 预算与票房关系
    fig2 = px.scatter(
        filtered_data, x='budget', y='revenue',
        title='预算与票房关系',
        labels={'budget': '预算 (美元)', 'revenue': '票房 (美元)'},
        trendline='ols',
        color='main_genre',
        hover_data=['title', 'release_year']
    )
    fig2.update_layout(
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font_color=font_color,
        xaxis={'gridcolor': grid_color},
        yaxis={'gridcolor': grid_color}
    )

    # 3. 年度票房趋势
    yearly_data = filtered_data.groupby('release_year').agg({'revenue': 'sum', 'budget': 'sum'}).reset_index()
    fig3 = make_subplots(specs=[[{'secondary_y': True}]])
    fig3.add_trace(
        go.Bar(x=yearly_data['release_year'], y=yearly_data['revenue'], name='年度总票房', marker_color='#2563eb'),
        secondary_y=False,
    )
    fig3.add_trace(
        go.Scatter(x=yearly_data['release_year'], y=yearly_data['budget'], name='年度总预算', marker_color='#ef4444'),
        secondary_y=True,
    )
    fig3.update_layout(
        title='年度票房与预算趋势',
        xaxis_title='年份',
        yaxis_title='票房 (美元)',
        yaxis2_title='预算 (美元)',
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font_color=font_color,
        xaxis={'gridcolor': grid_color},
        yaxis={'gridcolor': grid_color},
        yaxis2={'gridcolor': grid_color}
    )

    # 4. 电影类型票房分析
    genre_data = filtered_data.groupby('main_genre').agg({'revenue': 'mean', 'budget': 'mean'}).reset_index()
    fig4 = make_subplots(specs=[[{'secondary_y': True}]])
    fig4.add_trace(
        go.Bar(x=genre_data['main_genre'], y=genre_data['revenue'], name='平均票房', marker_color='#2563eb'),
        secondary_y=False,
    )
    fig4.add_trace(
        go.Scatter(x=genre_data['main_genre'], y=genre_data['budget'], name='平均预算', marker_color='#ef4444'),
        secondary_y=True,
    )
    fig4.update_layout(
        title='不同类型电影票房与预算分析',
        xaxis_title='电影类型',
        yaxis_title='平均票房 (美元)',
        yaxis2_title='平均预算 (美元)',
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font_color=font_color,
        xaxis={
            'gridcolor': grid_color, 
            'tickangle': 45
        },
        yaxis={'gridcolor': grid_color},
        yaxis2={'gridcolor': grid_color}
    )

    # 5. 评分与票房关系
    fig5 = px.scatter(
        filtered_data, x='vote_average', y='revenue',
        title='评分与票房关系',
        labels={'vote_average': '平均评分', 'revenue': '票房 (美元)'},
        trendline='ols',
        color='main_genre',
        hover_data=['title', 'release_year']
    )
    fig5.update_layout(
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font_color=font_color,
        xaxis={'gridcolor': grid_color},
        yaxis={'gridcolor': grid_color}
    )

    # 6. 导演影响力
    director_data = filtered_data.groupby('director').agg({'revenue': 'mean', 'vote_average': 'mean', 'id': 'count'}).reset_index()
    director_data = director_data.sort_values('revenue', ascending=False).head(15)
    fig6 = px.bar(
        director_data, x='director', y='revenue',
        title='导演平均票房排名 (前15名)',
        labels={'director': '导演', 'revenue': '平均票房 (美元)'},
        color='vote_average',
        hover_data=['vote_average', 'id']
    )
    fig6.update_layout(
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font_color=font_color,
        xaxis={'gridcolor': grid_color, 'tickangle': 45},
        yaxis={'gridcolor': grid_color}
    )

    # 7. 更新表格数据
    table_data = filtered_data.head(100).to_dict('records')

    return fig1, fig2, fig3, fig4, fig5, fig6, table_data

# 运行应用
if __name__ == '__main__':
    app.run(debug=True, port=8050)
'''

        with open(os.path.join(self.charts_dir, 'movie_box_office_dashboard.py'), 'w', encoding='utf-8') as f:
            f.write(dashboard_code)

        self.logger.info("交互式仪表板代码已保存")
        self.logger.info("运行以下命令启动仪表板:")
        self.logger.info(f"python {os.path.join(self.charts_dir, 'movie_box_office_dashboard.py')}")

        return True

    def run_complete_visualization(self, filename="cleaned_movie_data.csv", chunk_size=None, sample_fraction=None, max_scatter_points=10000):
        """运行完整的可视化流程"""
        self.logger.info("=" * 50)
        self.logger.info("开始完整的可视化流程")
        self.logger.info("=" * 50)

        # 1. 加载数据
        data = self.load_data(filename, chunk_size=chunk_size, sample_fraction=sample_fraction)
        if data is None:
            return None

        self.logger.info(f"加载的数据行数: {len(data)}")

        # 2. 生成基础可视化
        self.basic_visualizations(data)

        # 3. 生成交互式可视化
        self.interactive_visualizations(data, max_points=max_scatter_points)

        # 4. 创建交互式仪表板
        self.create_dashboard(data)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("可视化流程完成！")
        self.logger.info(f"图表已保存到: {self.charts_dir}")
        self.logger.info("=" * 50)

        return True


def main():
    """主函数，执行完整的可视化流程"""
    import argparse

    parser = argparse.ArgumentParser(description='电影票房数据可视化')
    parser.add_argument('--filename', default='cleaned_movie_data.csv', help='数据文件名')
    parser.add_argument('--chunk-size', type=int, help='分块加载大小')
    parser.add_argument('--sample-fraction', type=float, help='数据采样比例 (0-1)')
    parser.add_argument('--max-scatter-points', type=int, default=10000, help='交互式散点图最大数据点数量')
    parser.add_argument('--dashboard', action='store_true', help='直接运行交互式仪表板')

    args = parser.parse_args()

    if args.dashboard:
        # 直接运行仪表板
        import subprocess
        import sys
        dashboard_path = os.path.join(project_root, 'results', 'charts', 'movie_box_office_dashboard.py')
        
        # 检查仪表板文件是否存在
        if not os.path.exists(dashboard_path):
            print(f"仪表板文件不存在: {dashboard_path}")
            print("请先运行可视化脚本生成仪表板文件:")
            print("python src/visualization.py")
            sys.exit(1)
        
        # 运行仪表板
        print("正在启动交互式仪表板...")
        print(f"访问地址: http://localhost:8050")
        print("按 Ctrl+C 停止仪表板")
        subprocess.run([sys.executable, dashboard_path])
    else:
        # 运行完整的可视化流程
        viz = MovieVisualization()
        viz.run_complete_visualization(
            filename=args.filename,
            chunk_size=args.chunk_size,
            sample_fraction=args.sample_fraction,
            max_scatter_points=args.max_scatter_points
        )


if __name__ == "__main__":
    main()
