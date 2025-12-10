import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SocialMediaAnalysis:
    def __init__(self, base_dir="../data"):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "processed")
        self.raw_dir = os.path.join(base_dir, "raw")
        os.makedirs(self.processed_dir, exist_ok=True)

        # API密钥配置
        self.twitter_api_key = None
        self.twitter_api_secret = None
        self.imdb_api_key = None

    def load_data(self, filename="cleaned_movie_data.csv"):
        """加载处理后的数据"""
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            print("请先运行数据预处理脚本: python data_preprocessing.py")
            return None

        print(f"正在加载数据: {filename}")
        data = pd.read_csv(file_path)
        print(f"数据形状: {data.shape}")
        return data

    def extract_actors_from_data(self, movies_df):
        """从电影数据中提取演员列表"""
        print("\n正在提取演员列表...")

        if "cast" not in movies_df.columns:
            print("错误: 电影数据集中缺少cast列")
            return None

        # 提取所有演员
        all_actors = set()

        for cast_str in movies_df["cast"].dropna():
            try:
                # 处理cast字符串，假设格式为简单的逗号分隔
                cast = cast_str.split(",")
                # 去除空格并添加到集合中
                for actor in cast:
                    actor = actor.strip()
                    if actor:
                        all_actors.add(actor)
            except Exception:
                continue

        print(f"共提取到{len(all_actors)}位演员")
        return list(all_actors)

    def estimate_actor_influence(self, actor_name):
        """估计演员的影响力（简化版本，实际应使用社交媒体API）"""
        # 注意：这是一个简化的实现，实际项目中应该使用Twitter/Instagram API
        # 这里我们使用一个模拟的影响力评分

        # 演员影响力评分范围：1-100
        import hashlib

        # 使用演员名字的哈希值生成一个相对稳定的模拟评分
        hash_val = int(hashlib.md5(actor_name.encode()).hexdigest(), 16) % 100 + 1

        return hash_val

    def add_actor_influence_features(self, movies_df):
        """为电影数据添加演员影响力特征"""
        print("\n正在添加演员影响力特征...")

        if "cast" not in movies_df.columns:
            print("警告: 电影数据集中缺少cast列，无法添加演员影响力特征")
            return movies_df

        # 计算每位演员的影响力评分
        actor_influence = {}

        for cast_str in tqdm(movies_df["cast"].dropna(), desc="计算演员影响力"):
            try:
                cast = [actor.strip() for actor in cast_str.split(",") if actor.strip()]
                for actor in cast:
                    if actor not in actor_influence:
                        actor_influence[actor] = self.estimate_actor_influence(actor)
            except Exception:
                continue

        # 为每部电影计算演员影响力特征
        def calculate_movie_actor_influence(cast_str):
            if pd.isna(cast_str):
                return {
                    "top_actor_influence": 0,
                    "avg_actor_influence": 0,
                    "total_actor_influence": 0,
                    "num_actors": 0
                }

            try:
                cast = [actor.strip() for actor in cast_str.split(",") if actor.strip()]
                if not cast:
                    return {
                        "top_actor_influence": 0,
                        "avg_actor_influence": 0,
                        "total_actor_influence": 0,
                        "num_actors": 0
                    }

                # 获取每位演员的影响力评分
                influence_scores = [actor_influence.get(actor, 0) for actor in cast]

                return {
                    "top_actor_influence": max(influence_scores),
                    "avg_actor_influence": sum(influence_scores) / len(influence_scores),
                    "total_actor_influence": sum(influence_scores),
                    "num_actors": len(cast)
                }
            except Exception:
                return {
                    "top_actor_influence": 0,
                    "avg_actor_influence": 0,
                    "total_actor_influence": 0,
                    "num_actors": 0
                }

        # 应用函数计算演员影响力特征
        actor_influence_features = movies_df["cast"].apply(calculate_movie_actor_influence).apply(pd.Series)

        # 将新特征合并到电影数据中
        movies_df = pd.concat([movies_df, actor_influence_features], axis=1)

        print("演员影响力特征添加完成！")
        return movies_df

    def add_social_media_sentiment(self, movies_df):
        """为电影数据添加社交媒体舆情特征"""
        print("\n正在添加社交媒体舆情特征...")

        # 注意：这是一个简化的实现，实际项目中应该使用Twitter API或其他社交媒体API

        # 模拟社交媒体舆情评分
        def generate_sentiment_score(title):
            if pd.isna(title):
                return 0.5

            import hashlib
            # 使用电影标题的哈希值生成一个相对稳定的模拟评分
            hash_val = int(hashlib.md5(title.encode()).hexdigest(), 16) % 1000
            sentiment_score = 0.3 + (hash_val / 1000) * 0.4  # 范围：0.3-0.7
            return sentiment_score

        # 为每部电影生成舆情评分
        movies_df["social_media_sentiment"] = movies_df["title"].apply(generate_sentiment_score)

        # 添加模拟的社交媒体讨论热度
        def generate_social_media_engagement(title):
            if pd.isna(title):
                return 0

            import hashlib
            hash_val = int(hashlib.md5(title.encode()).hexdigest(), 16) % 1000000
            return hash_val

        movies_df["social_media_engagement"] = movies_df["title"].apply(generate_social_media_engagement)

        print("社交媒体舆情特征添加完成！")
        return movies_df

    def run_social_media_analysis(self, data):
        """运行完整的社交媒体分析流程"""
        print("=" * 50)
        print("开始社交媒体分析流程")
        print("=" * 50)

        # 1. 添加演员影响力特征
        data = self.add_actor_influence_features(data)

        # 2. 添加社交媒体舆情特征
        data = self.add_social_media_sentiment(data)

        # 3. 保存结果
        results_path = os.path.join(self.processed_dir, "movie_data_with_social_features.csv")
        data.to_csv(results_path, index=False)
        print(f"\n社交媒体分析结果已保存到: {results_path}")

        print("\n" + "=" * 50)
        print("社交媒体分析流程完成！")
        print("=" * 50)

        return data

    def main(self):
        """主函数"""
        # 1. 加载数据
        data = self.load_data()
        if data is None:
            return

        # 2. 运行社交媒体分析
        results = self.run_social_media_analysis(data)

        # 3. 显示结果示例
        if results is not None:
            print("\n社交媒体分析结果示例：")
            social_features = [
                "top_actor_influence", "avg_actor_influence", 
                "total_actor_influence", "num_actors",
                "social_media_sentiment", "social_media_engagement"
            ]
            print(results[social_features].head(5))

def main():
    """主函数"""
    sma = SocialMediaAnalysis()
    sma.main()

if __name__ == "__main__":
    main()
