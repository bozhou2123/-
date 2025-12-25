#!/usr/bin/env python3
"""
股票数据分析流水线主脚本
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import DataCollector
from src.analysis import EnhancedAnalysis
from src.generate_dashboard import DashboardGenerator

def setup_logging():
    """设置日志配置"""
    log_dir = "logs"
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f"{log_dir}/analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始股票数据分析流水线")
    
    try:
        # 1. 数据收集
        logger.info("阶段1: 数据收集")
        collector = DataCollector()
        
        # 加载股票列表
        stock_list = collector.load_stock_list()
        if stock_list is None:
            logger.error("无法加载股票列表，退出流程")
            return
        
        # 获取股票数据
        stock_codes = list(zip(stock_list['sector'], stock_list['code1']))
        stock_data = collector.fetch_stock_data(stock_codes)
        
        if not stock_data:
            logger.error("未获取到股票数据，退出流程")
            return
        
        # 保存数据
        collector.save_data(stock_data)
        
        # 2. 数据分析
        logger.info("阶段2: 数据分析")
        analyzer = EnhancedAnalysis()
        analysis_results = analyzer.run_full_analysis(stock_data)
        
        if not analysis_results:
            logger.error("分析失败，退出流程")
            return
        
        # 3. 生成仪表板
        logger.info("阶段3: 生成仪表板")
        dashboard_gen = DashboardGenerator()
        
        stock_features = None
        if hasattr(analyzer, 'stock_features'):
            stock_features = analyzer.stock_features
        
        dashboard_path = dashboard_gen.generate_dashboard(analysis_results, stock_features)
        
        logger.info(f"流水线执行完成，仪表板已生成: {dashboard_path}")
        
        # 4. 清理旧文件（保留最近7天的数据）
        cleanup_old_files()
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}", exc_info=True)
        sys.exit(1)

def cleanup_old_files(days_to_keep=7):
    """清理旧文件"""
    import glob
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    # 清理旧的数据文件
    data_files = glob.glob("data/processed/stock_data_*.pkl")
    for file_path in data_files:
        try:
            file_date_str = file_path.split('_')[-1].split('.')[0]
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            
            if file_date < cutoff_date:
                os.remove(file_path)
        except:
            pass
    
    # 清理旧的报告文件
    report_files = glob.glob("output/reports/*.json") + glob.glob("output/reports/*.csv")
    for file_path in report_files:
        try:
            if "latest" not in file_path:
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    os.remove(file_path)
        except:
            pass

if __name__ == "__main__":
    main()
