import pandas as pd
import numpy as np
import requests
import time
import yaml
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging

class DataCollector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_config = self.config['api']
        self.data_config = self.config['data']
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            "data/raw",
            "data/processed",
            "output/charts",
            "output/reports",
            "docs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_stock_list(self):
        """加载股票列表"""
        try:
            df = pd.read_excel(self.data_config['stock_list'])
            required_cols = ['sector', 'code1']
            
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"缺少必要的列: {col}")
            
            self.logger.info(f"成功加载 {len(df)} 只股票")
            return df
            
        except Exception as e:
            self.logger.error(f"加载股票列表失败: {e}")
            return None
    
    def fetch_stock_data(self, stock_codes, max_retries=3):
        """获取股票数据"""
        all_data = {}
        
        for idx, (sector, code) in enumerate(stock_codes):
            try:
                for attempt in range(max_retries):
                    try:
                        # 历史数据
                        history_url = f"{self.api_config['base_url']}{self.api_config['endpoints']['history']}"
                        history_url = f"{history_url}/d/n/{code}/{self.api_config['licence']}"
                        history_url += f"?st={self.data_config['start_date']}&et={self._get_end_date()}"
                        
                        response = requests.get(history_url, timeout=10)
                        
                        if response.status_code == 200:
                            history_data = self._parse_response(response.json(), code, sector)
                            
                            # 市值数据
                            market_url = f"{self.api_config['base_url']}{self.api_config['endpoints']['instrument']}"
                            market_url = f"{market_url}/{code}/{self.api_config['licence']}"
                            
                            response_market = requests.get(market_url, timeout=10)
                            market_data = {}
                            
                            if response_market.status_code == 200:
                                market_data = response_market.json()
                            
                            all_data[code] = {
                                'history': history_data,
                                'market': market_data,
                                'sector': sector
                            }
                            
                            self.logger.info(f"成功获取 {code} 数据")
                            time.sleep(0.2)  # 避免请求过快
                            break
                            
                    except requests.exceptions.RequestException as e:
                        self.logger.warning(f"第{attempt+1}次尝试获取 {code} 数据失败: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                        else:
                            self.logger.error(f"获取 {code} 数据失败")
                
            except Exception as e:
                self.logger.error(f"处理 {code} 时出错: {e}")
                continue
            
            # 进度显示
            if (idx + 1) % 20 == 0:
                self.logger.info(f"进度: {idx + 1}/{len(stock_codes)}")
        
        return all_data
    
    def _parse_response(self, response_data, code, sector):
        """解析API响应"""
        if isinstance(response_data, dict) and 'data' in response_data:
            data = response_data['data']
        else:
            data = response_data
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # 字段映射
        field_mapping = {
            't': 'date', 'o': 'open', 'h': 'high', 
            'l': 'low', 'c': 'close', 'v': 'volume',
            'a': 'amount', 'pc': 'pre_close'
        }
        
        # 重命名列
        for api_field, our_field in field_mapping.items():
            if api_field in df.columns:
                df[our_field] = df[api_field]
        
        # 添加元数据
        df['stock_code'] = code
        df['sector'] = sector
        
        # 转换日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 计算收益率
            df['daily_return'] = df['close'].pct_change()
        
        return df
    
    def _get_end_date(self):
        """获取结束日期"""
        return datetime.now().strftime("%Y%m%d")
    
    def save_data(self, data, filename=None):
        """保存数据"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_{timestamp}.pkl"
        
        filepath = f"data/processed/{filename}"
        
        try:
            # 保存为pickle格式
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"数据已保存到 {filepath}")
            
            # 同时保存最新的数据副本
            latest_path = "data/processed/latest_stock_data.pkl"
            with open(latest_path, 'wb') as f:
                pickle.dump(data, f)
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
