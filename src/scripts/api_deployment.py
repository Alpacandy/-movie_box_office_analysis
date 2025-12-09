import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import argparse
import uvicorn
import logging

# 初始化日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('api_deployment')

# 设置正确的项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(project_root)

# 修正：2. 将Pydantic模型定义提到模块级别
class MovieFeatures(BaseModel):
    budget: float
    popularity: float
    runtime: float
    vote_count: float
    vote_average: float
    release_year: int
    genre_action: int = 0
    genre_adventure: int = 0
    genre_animation: int = 0
    genre_comedy: int = 0
    genre_crime: int = 0
    genre_documentary: int = 0
    genre_drama: int = 0
    genre_family: int = 0
    genre_fantasy: int = 0
    genre_history: int = 0
    genre_horror: int = 0
    genre_music: int = 0
    genre_mystery: int = 0
    genre_romance: int = 0
    genre_science_fiction: int = 0
    genre_thriller: int = 0
    genre_war: int = 0
    genre_western: int = 0


class MovieBoxOfficeAPI:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(project_root, "results/models")
        # 修正：3. 使用正确的类名（注意大小写）
        self.app = FastAPI(
            title="电影票房预测API",
            description="基于机器学习模型的电影票房预测服务",
            version="1.0.0"
        )
        self.models_dir = models_dir
        self.models = {}
        self.load_models()
        self.setup_routes()

    def load_models(self):
        """加载所有可用的训练模型"""
        logger.info("正在加载训练模型...")

        if not os.path.exists(self.models_dir):
            logger.warning(f"模型目录不存在: {self.models_dir}")
            return

        for filename in os.listdir(self.models_dir):
            if filename.endswith("_model.joblib"):
                model_path = os.path.join(self.models_dir, filename)
                model_name = filename.replace("_model.joblib", "")

                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    logger.info(f"成功加载模型: {model_name}")
                except Exception as e:
                    logger.error(f"加载模型{model_name}失败: {e}")

    def setup_routes(self):
        """设置API路由"""
        app = self.app

        @app.get("/")
        def root():
            return {
                "message": "欢迎使用电影票房预测API",
                "models": list(self.models.keys()),
                "endpoints": [
                    "/predict - POST 电影票房预测",
                    "/models - GET 可用模型列表",
                    "/batch_predict - POST 批量预测"
                ]
            }

        @app.get("/models")
        def get_models():
            """获取可用模型列表"""
            return {
                "available_models": list(self.models.keys()),
                "total_models": len(self.models)
            }

        @app.post("/predict")
        def predict_box_office(features: MovieFeatures, model_name: str = "random_forest"):
            """预测电影票房"""
            if model_name not in self.models:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的模型名称: {model_name}。可用模型: {list(self.models.keys())}"
                )

            model = self.models[model_name]

            # 修正：4. 兼容Pydantic v2和v1
            features_dict = features.model_dump() if hasattr(features, 'model_dump') else features.dict()
            features_df = pd.DataFrame([features_dict])

            try:
                prediction = model.predict(features_df)

                return {
                    "success": True,
                    "model_name": model_name,
                    "features": features_dict,
                    "prediction": {
                        "revenue": float(prediction[0]),
                        "unit": "美元"
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

        @app.post("/batch_predict")
        def batch_predict_box_office(movies: List[MovieFeatures], model_name: str = "random_forest"):
            """批量预测电影票房"""
            if model_name not in self.models:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的模型名称: {model_name}。可用模型: {list(self.models.keys())}"
                )

            model = self.models[model_name]

            # 修正：5. 批量处理时使用同样的兼容性方案
            features_list = [
                movie.model_dump() if hasattr(movie, 'model_dump') else movie.dict()
                for movie in movies
            ]
            features_df = pd.DataFrame(features_list)

            try:
                predictions = model.predict(features_df)

                return {
                    "success": True,
                    "model_name": model_name,
                    "total_predictions": len(predictions),
                    "results": [
                        {
                            "features": features,
                            "prediction": {
                                "revenue": float(pred),
                                "unit": "美元"
                            }
                        }
                        for features, pred in zip(features_list, predictions)
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """启动API服务"""
        logger.info("\n" + "=" * 50)
        logger.info("电影票房预测API")
        logger.info("=" * 50)
        logger.info(f"API文档地址: http://{host}:{port}/docs")
        logger.info(f"模型列表: {list(self.models.keys())}")
        logger.info("=" * 50)

        uvicorn.run(self.app, host=host, port=port, debug=debug)


# 修正：6. 修复main函数的所有语法错误
def main():
    """主函数"""
    # 修正：7. 使用正确的类名和路径格式
    api = MovieBoxOfficeAPI(models_dir="../results/models")

    # 修正：8. 使用英文半角逗号和正确的括号
    parser = argparse.ArgumentParser(description="电影票房预测API服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()
    api.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()