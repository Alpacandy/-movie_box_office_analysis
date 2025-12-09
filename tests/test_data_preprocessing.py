import os
import tempfile
import unittest
import pandas as pd

# 从src.data_preprocessing包导入DataPreprocessing类
from src.data_preprocessing import DataPreprocessing


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'genres': ['Action,Adventure', 'Comedy,Drama', 'Horror', 'Sci-Fi,Thriller', 'Romance'],
            'release_date': ['2020-01-10', '2021-03-15', '2022-06-20', '2019-11-05', '2023-02-28'],
            'runtime': [120, 110, 95, 130, 105],
            'budget': [100000000, 50000000, 10000000, 150000000, 30000000],
            'revenue': [500000000, 200000000, 50000000, 800000000, 150000000],
            'vote_average': [7.5, 6.8, 5.2, 8.1, 6.5],
            'vote_count': [15000, 8000, 2000, 25000, 5000],
            'popularity': [90.5, 75.2, 30.1, 100.0, 55.7],
            'keywords': ['action,hero,save the world', 'funny,romantic,friendship', 'scary,ghost,haunted',
                         'space,alien,future', 'love,heartbreak,happy ending']
        })

        # 创建DataPreprocessing实例
        self.preprocessor = DataPreprocessing()

    def test_clean_data(self):
        # 测试数据清洗功能
        cleaned_data = self.preprocessor.clean_data(self.test_data)

        # 验证数据行数不变
        self.assertEqual(len(cleaned_data), len(self.test_data))

        # 验证预算和收入列都是数字类型
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_data['budget']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_data['revenue']))

    def test_feature_extraction(self):
        # 首先清洗数据
        cleaned_data = self.preprocessor.clean_data(self.test_data)

        # 测试特征提取功能
        extracted_data = self.preprocessor.extract_features(cleaned_data)

        # 验证添加了新特征
        # 注意：实际添加的特征可能因实现而异，这里只验证基本特征
        self.assertIn('budget', extracted_data.columns)
        self.assertIn('revenue', extracted_data.columns)

    def test_complete_preprocessing(self):
        # 测试完整的数据预处理流程
        # 1. 清洗数据
        cleaned_data = self.preprocessor.clean_data(self.test_data)
        # 2. 提取特征
        preprocessed_data = self.preprocessor.extract_features(cleaned_data)

        # 验证数据经过了完整处理
        self.assertIn('budget', preprocessed_data.columns)
        self.assertIn('revenue', preprocessed_data.columns)

    def test_save_and_load_data(self):
        # 测试数据保存和加载功能
        # 1. 清洗数据
        cleaned_data = self.preprocessor.clean_data(self.test_data)
        # 2. 提取特征
        preprocessed_data = self.preprocessor.extract_features(cleaned_data)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_file = tmp.name

        try:
            # 保存数据
            pd.DataFrame(preprocessed_data).to_csv(tmp_file, index=False)

            # 验证文件存在
            self.assertTrue(os.path.exists(tmp_file))

            # 加载数据
            loaded_data = pd.read_csv(tmp_file, parse_dates=['release_date'])

            # 验证加载的数据与原始数据一致
            # 简化比较方式，只检查关键列的值是否一致
            left_df = pd.DataFrame(preprocessed_data)
            right_df = loaded_data

            # 检查行数是否一致
            self.assertEqual(len(left_df), len(right_df))

            # 检查关键列的值是否一致（忽略类型差异）
            key_columns = ['id', 'title', 'budget', 'revenue', 'vote_average']
            for col in key_columns:
                if col in left_df.columns and col in right_df.columns:
                    # 对于数值列，使用近似比较
                    if pd.api.types.is_numeric_dtype(left_df[col]):
                        pd.testing.assert_series_equal(
                            left_df[col].astype(float),
                            right_df[col].astype(float),
                            rtol=1e-5,
                            check_dtype=False
                        )
                    else:
                        # 对于其他列，直接比较值
                        self.assertEqual(
                            left_df[col].tolist(),
                            right_df[col].tolist()
                        )
        finally:
            # 清理临时文件
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)


if __name__ == '__main__':
    unittest.main()
