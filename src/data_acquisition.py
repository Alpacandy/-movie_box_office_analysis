import os
import sys
import pandas as pd
import subprocess
import requests
from tqdm import tqdm
import time

# 获取当前脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, ".."))
# 添加项目根目录到sys.path
sys.path.append(project_root)

# 导入日志配置
from src.utils.logging_config import setup_logging, get_logger

# 配置日志系统
setup_logging(log_dir=os.path.join(project_root, "logs"))

class DataAcquisition:
    def __init__(self, base_dir=None):
        self.base_dir = os.path.join(project_root, "data") if base_dir is None else base_dir
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # 初始化日志系统
        self.logger = get_logger('data_acquisition')

        # API密钥配置
        self.tmdb_api_key = "34b35102e0c09331b0185f34e3c0f03e"  # 公共测试密钥，建议替换为自己的密钥
        self.omdb_api_key = None  # OMDB API密钥，用于获取IMDb数据

        # 支持的数据源
        self.supported_sources = ["tmdb", "pandas", "tmdb_api", "imdb", "box_office_mojo"]

    def download_tmdb_dataset(self):
        """从Kaggle下载TMDB 5000 Movie Dataset"""
        self.logger.info("正在下载TMDB 5000 Movie Dataset...")

        # 检查kaggle命令是否可用
        try:
            subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        except FileNotFoundError:
            self.logger.error("错误: 未找到kaggle命令行工具")
            self.logger.error("请先安装kaggle包: pip install kaggle")
            self.logger.error("然后配置Kaggle API密钥")
            self.logger.error("手动下载地址: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata ")
            return False
        except subprocess.CalledProcessError:
            self.logger.error("错误: kaggle命令执行失败")
            self.logger.error("请检查kaggle包是否正确安装")
            self.logger.error("手动下载地址: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata ")
            return False

        # 使用Kaggle API下载数据集
        try:
            subprocess.run([
                "kaggle", "datasets", "download", "-d", "tmdb/tmdb-movie-metadata", 
                "--path", self.raw_dir, "--unzip"
            ], check=True, capture_output=True)
            self.logger.info("TMDB数据集下载完成！")
            return True
        except subprocess.CalledProcessError:
            self.logger.error("Kaggle API下载失败")
            self.logger.error("请确保已正确配置Kaggle API密钥")
            self.logger.error("手动下载地址: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata ")
            return False

    def download_pandas_movie_dataset(self):
        """从MovieLens下载电影数据集（替代原pandas数据源）"""
        self.logger.info("正在下载MovieLens电影数据集...")

        # 使用MovieLens 1M数据集作为替代
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        file_path = os.path.join(self.raw_dir, "ml-1m.zip")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # 显示下载进度
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()

            self.logger.info("MovieLens数据集下载完成！")

            # 解压数据集
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            self.logger.info("MovieLens数据集解压完成！")

            # 将movies.dat转换为csv格式
            movies_dat_path = os.path.join(self.raw_dir, "ml-1m", "movies.dat")
            movies_csv_path = os.path.join(self.raw_dir, "movies.csv")

            # 读取movies.dat文件并转换为csv
            import pandas as pd
            movies = pd.read_csv(movies_dat_path, sep='::', engine='python', 
                                names=['movie_id', 'title', 'genres'], encoding='latin-1')
            movies.to_csv(movies_csv_path, index=False)
            self.logger.info("MovieLens数据集转换为csv格式完成！")

            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"下载失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            return False

    def load_tmdb_data(self):
        """加载TMDB数据集"""
        movies_path = os.path.join(self.raw_dir, "tmdb_5000_movies.csv")
        credits_path = os.path.join(self.raw_dir, "tmdb_5000_credits.csv")

        if not os.path.exists(movies_path) or not os.path.exists(credits_path):
            self.logger.error("TMDB数据文件不存在，请先运行download_tmdb_dataset()")
            return None, None

        self.logger.info("正在加载TMDB数据...")
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)

        self.logger.info(f"电影数据形状: {movies.shape}")
        self.logger.info(f"演职员数据形状: {credits.shape}")

        return movies, credits

    def load_pandas_movie_data(self):
        """加载pandas电影数据集"""
        file_path = os.path.join(self.raw_dir, "movies.csv")

        if not os.path.exists(file_path):
            print("pandas电影数据文件不存在，请先运行download_pandas_movie_dataset()")
            return None

        print("正在加载pandas电影数据...")
        movies = pd.read_csv(file_path)
        print(f"pandas电影数据形状: {movies.shape}")

        return movies

    def merge_tmdb_data(self, movies, credits):
        """合并TMDB电影数据和演职员数据"""
        print("正在合并TMDB数据...")

        # 重命名credits的列名以匹配movies
        credits.columns = ['id', 'title', 'cast', 'crew']

        # 合并数据集
        merged_data = movies.merge(credits, on='id')

        print(f"合并后数据形状: {merged_data.shape}")
        return merged_data

    def merge_multiple_datasets(self, tmdb_data, pandas_data):
        """合并多个数据集"""
        print("正在合并多个数据集...")

        # 确定tmdb_data中的标题列
        tmdb_title_col = 'title'
        if tmdb_title_col not in tmdb_data.columns:
            # 尝试使用title_x或title_y
            if 'title_x' in tmdb_data.columns:
                tmdb_title_col = 'title_x'
            elif 'title_y' in tmdb_data.columns:
                tmdb_title_col = 'title_y'
            else:
                # 如果没有找到title列，尝试使用original_title
                if 'original_title' in tmdb_data.columns:
                    tmdb_title_col = 'original_title'
                else:
                    print("错误: tmdb_data中没有找到标题列")
                    return tmdb_data

        # 确保标题列是字符串类型，并处理NaN值
        tmdb_data['title_lower'] = tmdb_data[tmdb_title_col].astype(str).fillna('').str.lower()
        pandas_data['title_lower'] = pandas_data['title'].astype(str).fillna('').str.lower()

        # 合并数据集
        merged_data = pd.merge(
            tmdb_data, 
            pandas_data, 
            left_on='title_lower', 
            right_on='title_lower', 
            how='inner',
            suffixes=('_tmdb', '_pandas')
        )

        # 删除临时列
        merged_data = merged_data.drop(['title_lower'], axis=1)

        print(f"多数据集合并后形状: {merged_data.shape}")
        return merged_data

    def get_latest_movies_from_tmdb(self, api_key=None, start_year=2017, end_year=None, max_pages=100):
        """从TMDB API获取2017年至今的电影数据"""
        import datetime

        # 如果未提供end_year，使用当前年份
        if end_year is None:
            end_year = datetime.datetime.now().year

        print(f"正在从TMDB API获取{start_year}-{end_year}年的电影数据...")

        # 如果没有提供API密钥，使用默认的公共密钥（仅用于演示，生产环境建议使用自己的密钥）
        if not api_key:
            print("警告: 未提供TMDB API密钥，使用默认公共密钥（可能有请求限制）")
            api_key = self.tmdb_api_key

        base_url = "https://api.themoviedb.org/3/discover/movie"  # 修复URL中的额外空格
        all_movies = []
        page = 1

        # 创建进度条
        pbar = tqdm(total=max_pages, desc="获取电影数据")

        while page <= max_pages:
            # 构建请求参数
            params = {
                "api_key": api_key,
                "language": "en-US",
                "sort_by": "release_date.desc",
                "include_adult": False,
                "include_video": False,
                "page": page,
                "primary_release_date.gte": f"{start_year}-01-01",
                "primary_release_date.lte": f"{end_year}-12-31"
            }

            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                # 提取电影数据
                movies = data.get("results", [])
                if not movies:
                    break

                # 为每部电影获取更详细的信息，包括票房和预算
                for movie in movies:
                    movie_id = movie["id"]
                    detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                    detail_params = {
                        "api_key": api_key,
                        "language": "en-US"
                    }
                    detail_response = requests.get(detail_url, params=detail_params)
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        # 添加票房和预算信息
                        movie["budget"] = detail_data.get("budget", 0)
                        movie["revenue"] = detail_data.get("revenue", 0)
                        movie["runtime"] = detail_data.get("runtime", 0)
                        movie["release_date"] = detail_data.get("release_date", "")

                    # 添加请求延迟，避免超过API速率限制
                    time.sleep(0.2)

                all_movies.extend(movies)

                # 检查是否还有更多页面
                total_pages = data.get("total_pages", 0)
                if page >= total_pages:
                    break

                page += 1
                pbar.update(1)

                # 添加请求延迟，避免超过API速率限制
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"\n错误: 获取电影数据失败: {e}")
                break

        pbar.close()

        if not all_movies:
            print("\n警告: 未获取到任何电影数据")
            return None

        # 转换为DataFrame
        movies_df = pd.DataFrame(all_movies)

        # 保存原始数据
        raw_file_path = os.path.join(self.raw_dir, f"tmdb_latest_movies_{start_year}_{end_year}.csv")
        movies_df.to_csv(raw_file_path, index=False)
        print(f"\n成功获取{len(movies_df)}部最新电影数据")
        print(f"原始数据已保存到: {raw_file_path}")

        # 返回处理后的数据
        return movies_df

    def get_imdb_data_by_title(self, title, api_key=None):
        """通过电影标题从OMDb API获取IMDb数据"""
        if not api_key and not self.omdb_api_key:
            print("警告: 未提供OMDb API密钥，无法获取IMDb数据")
            print("请注册OMDb API密钥: http://www.omdbapi.com/apikey.aspx ")
            return None

        omdb_key = api_key or self.omdb_api_key
        url = f"http://www.omdbapi.com/?t={title}&apikey={omdb_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data.get("Response") == "False":
                return None

            return data
        except requests.exceptions.RequestException as e:
            print(f"错误: 获取IMDb数据失败: {e}")
            return None

    def enrich_data_with_imdb(self, movies_df, api_key=None):
        """使用IMDb数据丰富电影数据集"""
        print("\n正在使用IMDb数据丰富电影数据集...")

        if "title" not in movies_df.columns:
            print("错误: 电影数据集中缺少title列")
            return movies_df

        # 准备IMDb数据列表
        imdb_data = []

        # 为每部电影获取IMDb数据
        for title in tqdm(movies_df["title"].unique(), desc="获取IMDb数据"):
            imdb_info = self.get_imdb_data_by_title(title, api_key)
            if imdb_info:
                imdb_data.append({
                    "title": title,
                    "imdb_id": imdb_info.get("imdbID"),
                    "imdb_rating": float(imdb_info.get("imdbRating", 0)) if imdb_info.get("imdbRating") != "N/A" else 0,
                    "imdb_votes": int(imdb_info.get("imdbVotes", "0").replace(",", "")) if imdb_info.get("imdbVotes") != "N/A" else 0,
                    "metascore": int(imdb_info.get("Metascore", "0")) if imdb_info.get("Metascore") != "N/A" else 0,
                    "imdb_genres": imdb_info.get("Genre"),
                    "imdb_director": imdb_info.get("Director"),
                    "imdb_actors": imdb_info.get("Actors"),
                    "imdb_plot": imdb_info.get("Plot")
                })

            # 添加请求延迟，避免超过API速率限制
            time.sleep(0.2)

        if not imdb_data:
            print("警告: 未获取到任何IMDb数据")
            return movies_df

        # 转换为DataFrame
        imdb_df = pd.DataFrame(imdb_data)

        # 保存IMDb数据
        imdb_file_path = os.path.join(self.raw_dir, "imdb_movie_data.csv")
        imdb_df.to_csv(imdb_file_path, index=False)
        print(f"IMDb数据已保存到: {imdb_file_path}")

        # 合并IMDb数据到电影数据集
        merged_df = pd.merge(movies_df, imdb_df, on="title", how="left")
        print(f"合并后数据形状: {merged_df.shape}")

        return merged_df

    def get_box_office_mojo_yearly_data(self, year=2024):
        """从Box Office Mojo获取年度票房数据（使用网络爬虫）"""
        print(f"\n正在从Box Office Mojo获取{year}年票房数据...")

        # Box Office Mojo年度票房URL
        url = f"https://www.boxofficemojo.com/year/world/{year}/"

        try:
            # 使用pandas读取HTML表格
            tables = pd.read_html(url)
            if not tables:
                print(f"错误: 无法从{url}获取数据")
                return None

            # 获取第一个表格（年度票房数据）
            bo_data = tables[0]

            # 保存Box Office Mojo数据
            bo_file_path = os.path.join(self.raw_dir, f"box_office_mojo_{year}.csv")
            bo_data.to_csv(bo_file_path, index=False)
            print(f"Box Office Mojo数据已保存到: {bo_file_path}")

            return bo_data
        except Exception as e:
            print(f"错误: 获取Box Office Mojo数据失败: {e}")
            print("请确保已安装pandas和lxml: pip install pandas lxml")
            return None

    def enrich_data_with_box_office_mojo(self, movies_df, year=2024):
        """使用Box Office Mojo数据丰富电影数据集"""
        print("\n正在使用Box Office Mojo数据丰富电影数据集...")

        if "title" not in movies_df.columns:
            print("错误: 电影数据集中缺少title列")
            return movies_df

        # 获取Box Office Mojo年度数据
        bo_data = self.get_box_office_mojo_yearly_data(year)
        if bo_data is None:
            return movies_df

        # 简化电影标题，去除年份和特殊字符，用于匹配
        def simplify_title(title):
            import re
            # 去除年份（如 "Movie (2024)" -> "Movie"）
            title = re.sub(r'\(\d{4}\)', '', title)
            # 去除多余空格和特殊字符
            title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
            title = title.strip().lower()
            return title

        # 添加简化标题列用于匹配
        movies_df["simplified_title"] = movies_df["title"].apply(simplify_title)
        bo_data["simplified_title"] = bo_data["Title"].apply(simplify_title)

        # 合并Box Office Mojo数据到电影数据集
        merged_df = pd.merge(movies_df, bo_data, on="simplified_title", how="left")

        # 删除临时列
        merged_df = merged_df.drop(["simplified_title"], axis=1)

        print(f"合并后数据形状: {merged_df.shape}")
        return merged_df

    def save_data(self, data, filename, processed=False):
        """保存数据到指定目录"""
        if processed:
            save_path = os.path.join(self.processed_dir, filename)
        else:
            save_path = os.path.join(self.raw_dir, filename)

        data.to_csv(save_path, index=False)
        self.logger.info(f"数据已保存到: {save_path}")

def main():
    """主函数，执行数据获取流程"""
    import sys

    # 添加命令行参数支持
    skip_download = '--skip-download' in sys.argv
    get_latest = '--get-latest' in sys.argv
    use_imdb = '--use-imdb' in sys.argv
    use_box_office_mojo = '--use-box-office-mojo' in sys.argv
    imdb_api_key = None

    # 检查是否提供了IMDb API密钥
    for arg in sys.argv:
        if arg.startswith('--imdb-api-key='):
            imdb_api_key = arg.split('=')[1]

    da = DataAcquisition()

    da.logger.info("=" * 50)
    da.logger.info("开始数据获取流程")
    da.logger.info("=" * 50)
    da.logger.info("使用选项:")
    da.logger.info("  --skip-download          跳过自动下载，直接使用现有数据")
    da.logger.info("  --get-latest             从TMDB API获取2024-2025年最新电影数据")
    da.logger.info("  --use-imdb               使用IMDb数据丰富电影数据集")
    da.logger.info("  --imdb-api-key=KEY       设置OMDb API密钥")
    da.logger.info("  --use-box-office-mojo    使用Box Office Mojo数据丰富电影数据集")
    da.logger.info("=" * 50)

    latest_movies = None

    if get_latest:
        # 从TMDB API获取2017年至今的最新电影数据
        da.logger.info("\n正在从TMDB API获取2017年至今最新电影数据...")
        latest_movies = da.get_latest_movies_from_tmdb(start_year=2017, end_year=None, max_pages=5)  # 限制为5页，避免请求过多
        if latest_movies is not None:
            da.logger.info("\n最新电影数据获取完成！")

            # 使用IMDb数据丰富最新电影数据
            if use_imdb:
                latest_movies = da.enrich_data_with_imdb(latest_movies, imdb_api_key)

            # 使用Box Office Mojo数据丰富最新电影数据
            if use_box_office_mojo:
                # 获取当前年份
                import datetime
                current_year = datetime.datetime.now().year
                # 为最近几年获取Box Office Mojo数据
                for year in range(current_year - 2, current_year + 1):
                    latest_movies = da.enrich_data_with_box_office_mojo(latest_movies, year=year)

            # 保存丰富后的最新电影数据
            import datetime
            current_year = datetime.datetime.now().year
            save_filename = f"tmdb_latest_movies_2017_{current_year}.csv"
            if use_imdb:
                save_filename = save_filename.replace(".csv", "_with_imdb.csv")
            if use_box_office_mojo:
                save_filename = save_filename.replace(".csv", "_with_bo_mojo.csv")
            da.save_data(latest_movies, save_filename)
    elif not skip_download:
        # 下载TMDB数据集
        da.logger.info("\n1. 正在尝试下载TMDB 5000 Movie Dataset...")
        da.download_tmdb_dataset()

        # 下载pandas电影数据集
        da.logger.info("\n2. 正在尝试下载pandas电影数据集...")
        da.download_pandas_movie_dataset()
    else:
        da.logger.info("\n跳过自动下载，将使用现有数据文件（如果存在）")

    da.logger.info("\n" + "=" * 50)
    da.logger.info("开始数据加载和合并")
    da.logger.info("=" * 50)

    # 加载数据
    movies_tmdb, credits_tmdb = da.load_tmdb_data()
    movies_pandas = da.load_pandas_movie_data()

    # 检查数据是否成功加载
    if movies_tmdb is None or credits_tmdb is None:
        da.logger.warning("\nTMDB数据加载失败，跳过TMDB数据处理")
        da.logger.warning("请检查data/raw目录下是否有tmdb_5000_movies.csv和tmdb_5000_credits.csv文件")
        da.logger.warning("如果没有，请手动下载TMDB 5000 Movie Dataset")
        da.logger.warning("手动下载地址: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata ")
    else:
        da.logger.info(f"\n成功加载TMDB数据：{len(movies_tmdb)}部电影")
        # 合并TMDB数据
        tmdb_merged = da.merge_tmdb_data(movies_tmdb, credits_tmdb)

        # 使用IMDb数据丰富TMDB合并数据
        if use_imdb:
            tmdb_merged = da.enrich_data_with_imdb(tmdb_merged, imdb_api_key)

        # 使用Box Office Mojo数据丰富TMDB合并数据
        if use_box_office_mojo:
            # 为旧数据尝试获取多个年份的Box Office Mojo数据
            for year in range(2010, 2026):
                tmdb_merged = da.enrich_data_with_box_office_mojo(tmdb_merged, year=year)

        da.save_data(tmdb_merged, "tmdb_merged.csv")

        if movies_pandas is not None:
                da.logger.info(f"成功加载pandas电影数据：{len(movies_pandas)}部电影")
                # 合并多个数据集
                final_merged = da.merge_multiple_datasets(tmdb_merged, movies_pandas)

                # 检查是否有MovieLens 1M数据集并合并
                ml_1m_dir = os.path.join(da.raw_dir, "ml-1m")
                if os.path.exists(ml_1m_dir):
                    da.logger.info("检测到MovieLens 1M数据集，开始合并...")
                    # 加载MovieLens 1M movies.dat文件
                    movies_dat_path = os.path.join(ml_1m_dir, "movies.dat")
                    if os.path.exists(movies_dat_path):
                        # MovieLens 1M movies.dat文件格式：MovieID::Title::Genres
                        ml_movies = pd.read_csv(movies_dat_path, sep='::', engine='python', 
                                             names=['ml_movie_id', 'ml_title', 'ml_genres'], 
                                             encoding='latin-1')

                        # 处理MovieLens电影标题，提取年份
                        ml_movies['ml_year'] = ml_movies['ml_title'].str.extract(r'\((\d{4})\)$', expand=False)
                        ml_movies['ml_title_clean'] = ml_movies['ml_title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()

                        # 处理评分数据
                        ratings_file = os.path.join(ml_1m_dir, 'ratings.dat')
                        if os.path.exists(ratings_file):
                            da.logger.info("加载MovieLens 1M评分数据...")
                            ml_ratings = pd.read_csv(ratings_file, sep='::', engine='python', 
                                                   names=['UserID', 'MovieID', 'Rating', 'Timestamp'], 
                                                   encoding='latin-1')

                            # 计算电影平均评分和评分数量
                            ml_rating_stats = ml_ratings.groupby('MovieID').agg({
                                'Rating': ['mean', 'count']
                            }).reset_index()
                            ml_rating_stats.columns = ['ml_movie_id', 'ml_avg_rating', 'ml_rating_count']

                            # 合并评分统计到电影数据
                            ml_movies = ml_movies.merge(ml_rating_stats, on='ml_movie_id', how='left')

                        # 合并到主数据集（使用标题匹配）
                        # 清洗主数据集的标题
                        final_merged['title_clean'] = final_merged['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip().str.lower()
                        ml_movies['ml_title_clean_lower'] = ml_movies['ml_title_clean'].str.lower()

                        # 合并数据集
                        final_merged = final_merged.merge(ml_movies, left_on='title_clean', right_on='ml_title_clean_lower', how='left')

                        # 删除临时列
                        final_merged = final_merged.drop(['title_clean', 'ml_title_clean_lower'], axis=1)

                        da.logger.info(f"MovieLens 1M数据集合并完成，总记录数: {len(final_merged)}")

                # 保存最终合并数据
                da.save_data(final_merged, "movies_combined.csv")
                da.logger.info("\n" + "=" * 50)
                da.logger.info("数据获取和合并完成！")
                da.logger.info("=" * 50)
        else:
            da.logger.warning("\n警告: pandas电影数据加载失败，跳过多数据集合并")
            da.logger.warning("仅保存TMDB合并数据")
            da.logger.info("\n" + "=" * 50)
            da.logger.info("TMDB数据处理完成，但pandas电影数据加载失败")
            da.logger.info("=" * 50)

    da.logger.info("\n" + "=" * 50)
    da.logger.info("数据获取流程结束")
    da.logger.info("=" * 50)
    da.logger.info("\n下一步操作:")
    da.logger.info("1. 数据预处理: python data_preprocessing.py")
    da.logger.info("2. 探索性数据分析: python eda_analysis.py")
    da.logger.info("3. 特征工程: python feature_engineering.py")
    da.logger.info("4. 模型训练: python modeling.py")
    da.logger.info("5. 深度学习模型: python deep_learning.py")
    da.logger.info("6. 可视化: python visualization.py")
    da.logger.info("7. 模型解释: python model_interpretability.py")
    da.logger.info("8. API部署: python api_deployment.py")
    da.logger.info("\n详细操作说明请参考: operation_manual.md")

if __name__ == "__main__":
    main()