import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from datetime import datetime
import json
import logging
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnalysis:
    def __init__(self, config_path="config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.analysis_config = self.config['analysis']
        self.output_config = self.config['output']
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据存储
        self.data = {}
        self.results = {}
    
    def run_full_analysis(self, stock_data):
        """运行完整分析流程"""
        self.logger.info("开始股票数据分析")
        
        # 1. 准备数据
        self._prepare_data(stock_data)
        
        # 2. 个股聚类分析
        self.logger.info("进行个股聚类分析")
        stock_clusters = self._analyze_stock_clusters()
        
        # 3. 板块关联分析
        self.logger.info("进行板块关联分析")
        sector_relations = self._analyze_sector_relations()
        
        # 4. 风险分析
        self.logger.info("进行风险分析")
        risk_metrics = self._calculate_risk_metrics()
        
        # 5. 生成投资建议
        self.logger.info("生成投资建议")
        recommendations = self._generate_recommendations()
        
        # 6. 保存结果
        results = {
            'stock_clusters': stock_clusters,
            'sector_relations': sector_relations,
            'risk_metrics': risk_metrics,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'stock_count': len(self.stock_features) if hasattr(self, 'stock_features') else 0
        }
        
        self.results = results
        self._save_results(results)
        
        return results
    
    def _prepare_data(self, stock_data):
        """准备分析数据"""
        self.data = stock_data
        
        # 提取历史数据
        self.history_data = {}
        self.market_data = {}
        
        for code, data_dict in stock_data.items():
            if 'history' in data_dict:
                df = data_dict['history']
                if len(df) > 10:  # 至少需要10个交易日的数据
                    self.history_data[code] = df
            
            if 'market' in data_dict:
                self.market_data[code] = data_dict['market']
        
        self.logger.info(f"准备分析 {len(self.history_data)} 只股票数据")
    
    def _analyze_stock_clusters(self):
        """分析个股聚类"""
        # 计算特征
        features = []
        
        for code, df in self.history_data.items():
            if len(df) < 20:
                continue
            
            try:
                # 计算波动率
                returns = df['daily_return'].dropna()
                annual_vol = returns.std() * np.sqrt(252/len(returns))
                
                # 获取市值
                market_cap = self._extract_market_cap(code)
                
                # 计算技术指标
                rsi = self._calculate_rsi(df)
                momentum = self._calculate_momentum(df)
                
                features.append({
                    'code': code,
                    'sector': df['sector'].iloc[0] if 'sector' in df.columns else '未知',
                    'market_cap': market_cap,
                    'log_market_cap': np.log(market_cap) if market_cap > 0 else 0,
                    'volatility': annual_vol,
                    'rsi': rsi,
                    'momentum': momentum,
                    'avg_volume': df['volume'].mean() if 'volume' in df.columns else 0
                })
                
            except Exception as e:
                self.logger.warning(f"计算 {code} 特征失败: {e}")
                continue
        
        if len(features) < 10:
            self.logger.warning("特征数据不足，无法进行聚类分析")
            return None
        
        self.stock_features = pd.DataFrame(features)
        
        # PCA降维
        numeric_cols = ['log_market_cap', 'volatility', 'rsi', 'momentum']
        X = self.stock_features[numeric_cols].fillna(0).values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means聚类
        best_k = self._find_optimal_clusters(X_scaled)
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        self.stock_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # 可视化
        self._plot_stock_clusters(X_scaled, kmeans)
        
        return {
            'features': self.stock_features.to_dict('records'),
            'cluster_summary': self.stock_features.groupby('cluster').agg({
                'code': 'count',
                'market_cap': ['mean', 'std'],
                'volatility': ['mean', 'std'],
                'sector': lambda x: x.mode()[0] if len(x.mode()) > 0 else '混合'
            }).round(2).to_dict(),
            'optimal_k': best_k
        }
    
    def _analyze_sector_relations(self):
        """分析板块关联性"""
        # 计算板块收益率
        sector_returns = {}
        
        for code, df in self.history_data.items():
            sector = df['sector'].iloc[0] if 'sector' in df.columns else '未知'
            
            if sector not in sector_returns:
                sector_returns[sector] = []
            
            returns = df['daily_return'].dropna()
            if len(returns) > 10:
                sector_returns[sector].append(returns)
        
        # 计算每个板块的平均收益率
        sector_avg_returns = {}
        for sector, returns_list in sector_returns.items():
            if returns_list:
                # 对齐收益率序列
                min_len = min([len(r) for r in returns_list])
                aligned = [r.iloc[-min_len:].reset_index(drop=True) for r in returns_list]
                
                # 计算平均收益率
                avg_returns = pd.concat(aligned, axis=1).mean(axis=1)
                sector_avg_returns[sector] = avg_returns
        
        if len(sector_avg_returns) < 3:
            self.logger.warning("板块数据不足，无法进行关联分析")
            return None
        
        # 创建收益率DataFrame
        sector_df = pd.DataFrame(sector_avg_returns)
        
        # 计算相关性矩阵
        correlation_matrix = sector_df.corr()
        
        # 格兰杰因果检验
        granger_results = self._perform_granger_tests(sector_df)
        
        # 可视化
        self._plot_sector_network(correlation_matrix)
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'granger_causality': granger_results,
            'sector_count': len(sector_avg_returns)
        }
    
    def _calculate_risk_metrics(self):
        """计算风险指标"""
        risk_metrics = {}
        
        for code, df in self.history_data.items():
            if len(df) < 20:
                continue
            
            try:
                returns = df['daily_return'].dropna()
                
                # VaR计算（95%置信度）
                var_95 = np.percentile(returns, 5)
                
                # CVaR计算
                cvar_95 = returns[returns <= var_95].mean()
                
                # 最大回撤
                cum_returns = (1 + returns).cumprod()
                peak = cum_returns.expanding().max()
                drawdown = (cum_returns - peak) / peak
                max_drawdown = drawdown.min()
                
                # 夏普比率（假设无风险利率为2%）
                sharpe_ratio = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
                
                risk_metrics[code] = {
                    'VaR_95': var_95,
                    'CVaR_95': cvar_95,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'volatility': returns.std() * np.sqrt(252)
                }
                
            except Exception as e:
                self.logger.warning(f"计算 {code} 风险指标失败: {e}")
                continue
        
        return risk_metrics
    
    def _generate_recommendations(self):
        """生成投资建议"""
        if not hasattr(self, 'stock_features'):
            return {}
        
        recommendations = {
            'high_quality': [],  # 高质量股票
            'high_risk': [],     # 高风险高收益
            'stable': [],        # 稳定收益
            'avoid': []          # 避免投资
        }
        
        for _, row in self.stock_features.iterrows():
            score = self._calculate_stock_score(row)
            
            if score >= 8:
                recommendations['high_quality'].append({
                    'code': row['code'],
                    'sector': row['sector'],
                    'score': score,
                    'reason': '低波动、高动量、市值适中'
                })
            elif score <= 3:
                recommendations['avoid'].append({
                    'code': row['code'],
                    'sector': row['sector'],
                    'score': score,
                    'reason': '高波动、动量不足'
                })
            elif row['volatility'] > 0.4:
                recommendations['high_risk'].append({
                    'code': row['code'],
                    'sector': row['sector'],
                    'score': score,
                    'reason': '高波动率，适合激进投资者'
                })
            elif row['volatility'] < 0.2:
                recommendations['stable'].append({
                    'code': row['code'],
                    'sector': row['sector'],
                    'score': score,
                    'reason': '低波动率，适合稳健投资者'
                })
        
        return recommendations
    
    def _calculate_stock_score(self, row):
        """计算股票综合评分（0-10分）"""
        score = 5  # 基础分
        
        # 波动率评分（越低越好）
        if row['volatility'] < 0.2:
            score += 2
        elif row['volatility'] > 0.4:
            score -= 2
        
        # RSI评分（接近50最好）
        if 40 < row['rsi'] < 60:
            score += 1
        
        # 动量评分
        if row['momentum'] > 0.1:
            score += 2
        elif row['momentum'] < -0.1:
            score -= 1
        
        # 市值评分（适中最好）
        market_cap = row['market_cap']
        if 1e9 < market_cap < 1e11:
            score += 1
        
        return min(max(score, 0), 10)
    
    def _save_results(self, results):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON
        json_path = f"{self.output_config['reports_dir']}/analysis_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV
        if hasattr(self, 'stock_features'):
            csv_path = f"{self.output_config['reports_dir']}/stock_features_{timestamp}.csv"
            self.stock_features.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存最新的结果
        latest_json = f"{self.output_config['reports_dir']}/latest_results.json"
        with open(latest_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"分析结果已保存")
    
    def _plot_stock_clusters(self, X_pca, kmeans):
        """绘制股票聚类图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # PCA散点图
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=kmeans.labels_, cmap='viridis', alpha=0.7)
        axes[0].set_title('个股PCA聚类结果')
        axes[0].set_xlabel('主成分1')
        axes[0].set_ylabel('主成分2')
        axes[0].grid(True, alpha=0.3)
        
        # 特征散点图
        if hasattr(self, 'stock_features'):
            scatter2 = axes[1].scatter(self.stock_features['log_market_cap'],
                                      self.stock_features['volatility'],
                                      c=kmeans.labels_, cmap='viridis', alpha=0.7)
            axes[1].set_title('市值与波动率关系')
            axes[1].set_xlabel('对数市值')
            axes[1].set_ylabel('年化波动率')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = f"{self.output_config['charts_dir']}/stock_clusters.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sector_network(self, correlation_matrix):
        """绘制板块关系网络图"""
        G = nx.Graph()
        
        # 添加节点
        for sector in correlation_matrix.columns:
            G.add_node(sector)
        
        # 添加边（相关性高于阈值）
        threshold = self.analysis_config['correlation_threshold']
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    sector1 = correlation_matrix.columns[i]
                    sector2 = correlation_matrix.columns[j]
                    weight = abs(corr)
                    color = 'red' if corr > 0 else 'blue'
                    
                    G.add_edge(sector1, sector2, weight=weight, color=color)
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制边
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              width=edge_weights, alpha=0.6)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                              node_size=800, alpha=0.8)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f'板块关联网络图（相关性≥{threshold}）')
        plt.axis('off')
        
        chart_path = f"{self.output_config['charts_dir']}/sector_network.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _find_optimal_clusters(self, X):
        """寻找最优聚类数量"""
        max_k = min(self.analysis_config['max_clusters'], len(X) // 5)
        if max_k < 2:
            return 2
        
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        if silhouette_scores:
            best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
            return best_k
        
        return 3
    
    def _perform_granger_tests(self, sector_df, maxlag=5):
        """执行格兰杰因果检验"""
        results = {}
        
        for i in range(len(sector_df.columns)):
            for j in range(len(sector_df.columns)):
                if i != j:
                    cause = sector_df.columns[i]
                    effect = sector_df.columns[j]
                    
                    try:
                        # 准备数据
                        data = sector_df[[cause, effect]].dropna()
                        
                        # 格兰杰检验
                        test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                        
                        # 提取最佳p值
                        p_values = []
                        for lag in range(1, maxlag + 1):
                            p_value = test_result[lag][0]['ssr_chi2test'][1]
                            p_values.append(p_value)
                        
                        min_p = min(p_values)
                        best_lag = p_values.index(min_p) + 1
                        
                        results[f"{cause}→{effect}"] = {
                            'p_value': min_p,
                            'best_lag': best_lag,
                            'causal': min_p < 0.05
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"格兰杰检验 {cause}→{effect} 失败: {e}")
                        continue
        
        return results
    
    def _extract_market_cap(self, code):
        """提取市值数据"""
        if code in self.market_data:
            market_data = self.market_data[code]
            
            # 尝试不同的市值字段
            market_cap_fields = ['market_cap', 'marketCap', 'marketValue', 
                                'totalMarketCap', 'circulationMarketValue']
            
            for field in market_cap_fields:
                if field in market_data:
                    value = market_data[field]
                    
                    if isinstance(value, (int, float)):
                        return value
                    elif isinstance(value, str):
                        # 处理字符串格式的市值
                        try:
                            if '亿' in value:
                                return float(value.replace('亿', '')) * 1e8
                            elif '万' in value:
                                return float(value.replace('万', '')) * 1e4
                            else:
                                return float(value)
                        except:
                            continue
        
        # 如果无法获取市值，使用默认值
        return 1e9
    
    def _calculate_rsi(self, df, period=14):
        """计算RSI指标"""
        if 'close' not in df.columns or len(df) < period + 1:
            return 50
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _calculate_momentum(self, df, period=10):
        """计算动量指标"""
        if 'close' not in df.columns or len(df) < period + 1:
            return 0
        
        momentum = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1)
        return momentum
