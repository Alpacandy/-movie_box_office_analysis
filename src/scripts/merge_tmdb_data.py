import os
import pandas as pd
import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tmdb_merger')

class TMDBDatasetMerger:
    def __init__(self, base_dir=None):
        # 如果没有指定基础目录，使用项目根目录下的data目录
        if base_dir is None:
            # 获取脚本所在目录的绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算项目根目录
            project_root = os.path.abspath(os.path.join(script_dir, '../..'))
            # 设置基础目录为项目根目录下的data目录
            self.base_dir = os.path.join(project_root, 'data')
        else:
            self.base_dir = base_dir

        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.movies_file = "tmdb_5000_movies.csv"
        self.credits_file = "tmdb_5000_credits.csv"
        self.output_file = "tmdb_merged.csv"

    def merge(self):
        """合并TMDB电影数据和演职员数据"""
        logger.info("=" * 50)
        logger.info("开始合并TMDB 5000电影数据集")
        logger.info("=" * 50)

        # 检查文件是否存在
        movies_path = os.path.join(self.raw_dir, self.movies_file)
        credits_path = os.path.join(self.raw_dir, self.credits_file)

        if not os.path.exists(movies_path):
            logger.error(f"文件不存在: {movies_path}")
            return False

        if not os.path.exists(credits_path):
            logger.error(f"文件不存在: {credits_path}")
            return False

        # 加载数据
        logger.info("正在加载电影数据...")
        movies = pd.read_csv(movies_path)
        logger.info(f"电影数据形状: {movies.shape}")

        logger.info("正在加载演职员数据...")
        credits = pd.read_csv(credits_path)
        logger.info(f"演职员数据形状: {credits.shape}")

        # 合并数据
        logger.info("正在合并数据...")

        # 重命名credits的列名以匹配movies
        credits.columns = ['id', 'title', 'cast', 'crew']

        # 合并数据集
        merged_data = movies.merge(credits, on='id')

        logger.info(f"合并后数据形状: {merged_data.shape}")

        # 保存合并后的数据
        output_path = os.path.join(self.raw_dir, self.output_file)
        merged_data.to_csv(output_path, index=False)
        logger.info(f"合并后的数据已保存到: {output_path}")

        logger.info("=" * 50)
        logger.info("TMDB数据集合并完成")
        logger.info("=" * 50)

        return True

if __name__ == "__main__":
    merger = TMDBDatasetMerger()
    merger.merge()
