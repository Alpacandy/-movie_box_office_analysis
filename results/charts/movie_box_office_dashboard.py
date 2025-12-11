# -*- coding: utf-8 -*-
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
