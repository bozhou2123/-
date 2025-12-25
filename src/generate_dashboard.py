import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path

class DashboardGenerator:
    def __init__(self, config_path="config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_config = self.config['output']
        self.template_dir = "templates"
        
        # 创建目录
        Path(self.template_dir).mkdir(exist_ok=True)
    
    def generate_dashboard(self, analysis_results, stock_features=None):
        """生成交互式HTML仪表板"""
        self.logger.info("开始生成仪表板")
        
        # 加载数据
        if isinstance(analysis_results, str):
            with open(analysis_results, 'r', encoding='utf-8') as f:
                results = json.load(f)
        else:
            results = analysis_results
        
        # 创建图表
        charts_html = self._create_charts(results, stock_features)
        
        # 创建指标卡片
        metrics_html = self._create_metrics_cards(results)
        
        # 创建股票表格
        tables_html = self._create_tables(results, stock_features)
        
        # 读取模板
        template_path = os.path.join(self.template_dir, "dashboard_template.html")
        
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        else:
            template = self._get_default_template()
        
        # 填充模板
        dashboard_html = template.replace("<!-- CHARTS_PLACEHOLDER -->", charts_html)
        dashboard_html = dashboard_html.replace("<!-- METRICS_PLACEHOLDER -->", metrics_html)
        dashboard_html = dashboard_html.replace("<!-- TABLES_PLACEHOLDER -->", tables_html)
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dashboard_html = dashboard_html.replace("{{LAST_UPDATE}}", timestamp)
        
        # 保存仪表板
        output_path = os.path.join(self.output_config['html_dir'], "index.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        # 创建历史数据页面
        self._create_history_page(results)
        
        self.logger.info(f"仪表板已生成: {output_path}")
        return output_path
    
    def _create_charts(self, results, stock_features):
        """创建图表"""
        charts_html = ""
        
        # 1. 股票聚类图
        if 'stock_clusters' in results and stock_features is not None:
            fig = self._create_cluster_chart(stock_features)
            charts_html += self._fig_to_html(fig, "股票聚类分析")
        
        # 2. 板块相关性热图
        if 'sector_relations' in results:
            fig = self._create_correlation_heatmap(results['sector_relations'])
            charts_html += self._fig_to_html(fig, "板块相关性热图")
        
        # 3. 风险指标分布图
        if 'risk_metrics' in results:
            fig = self._create_risk_distribution_chart(results['risk_metrics'])
            charts_html += self._fig_to_html(fig, "风险指标分布")
        
        # 4. 投资建议分布图
        if 'recommendations' in results:
            fig = self._create_recommendation_chart(results['recommendations'])
            charts_html += self._fig_to_html(fig, "投资建议分布")
        
        return charts_html
    
    def _create_metrics_cards(self, results):
        """创建指标卡片"""
        cards_html = ""
        
        metrics = [
            {
                'title': '分析股票数量',
                'value': results.get('stock_count', 0),
                'icon': '📊',
                'color': 'primary'
            },
            {
                'title': '板块数量',
                'value': results.get('sector_relations', {}).get('sector_count', 0),
                'icon': '🏢',
                'color': 'success'
            },
            {
                'title': '高评分股票',
                'value': len(results.get('recommendations', {}).get('high_quality', [])),
                'icon': '⭐',
                'color': 'warning'
            },
            {
                'title': '风险提示',
                'value': len(results.get('recommendations', {}).get('avoid', [])),
                'icon': '⚠️',
                'color': 'danger'
            }
        ]
        
        for metric in metrics:
            card = f"""
            <div class="col-md-3 mb-4">
                <div class="card border-{metric['color']} h-100">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="card-subtitle mb-2 text-muted">{metric['title']}</h6>
                                <h2 class="card-title">{metric['icon']} {metric['value']}</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
            cards_html += card
        
        return cards_html
    
    def _create_tables(self, results, stock_features):
        """创建数据表格"""
        tables_html = ""
        
        # 1. 优质股票推荐表
        if 'recommendations' in results and results['recommendations'].get('high_quality'):
            table_data = []
            for stock in results['recommendations']['high_quality'][:10]:
                table_data.append([
                    stock['code'],
                    stock['sector'],
                    f"{stock['score']}/10",
                    stock['reason']
                ])
            
            df = pd.DataFrame(table_data, columns=['股票代码', '板块', '评分', '推荐理由'])
            tables_html += self._df_to_html_table(df, "优质股票推荐")
        
        # 2. 股票特征表
        if stock_features is not None:
            df_sample = stock_features[['code', 'sector', 'cluster', 'volatility', 'market_cap']].head(10)
            df_sample['market_cap'] = df_sample['market_cap'].apply(lambda x: f"{x/1e8:.2f}亿")
            tables_html += self._df_to_html_table(df_sample, "股票特征示例")
        
        return tables_html
    
    def _create_cluster_chart(self, stock_features):
        """创建聚类图"""
        fig = px.scatter(
            stock_features,
            x='log_market_cap',
            y='volatility',
            color='cluster',
            hover_data=['code', 'sector', 'market_cap'],
            title='股票聚类分析',
            labels={
                'log_market_cap': '对数市值',
                'volatility': '年化波动率',
                'cluster': '聚类'
            }
        )
        
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        fig.update_layout(height=500)
        
        return fig
    
    def _create_correlation_heatmap(self, sector_relations):
        """创建相关性热图"""
        if 'correlation_matrix' not in sector_relations:
            return go.Figure()
        
        corr_matrix = pd.DataFrame(sector_relations['correlation_matrix'])
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='板块收益率相关性矩阵',
            height=500,
            xaxis_title="板块",
            yaxis_title="板块"
        )
        
        return fig
    
    def _create_risk_distribution_chart(self, risk_metrics):
        """创建风险分布图"""
        if not risk_metrics:
            return go.Figure()
        
        # 提取风险数据
        volatilities = [m['volatility'] for m in risk_metrics.values()]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('波动率分布', '最大回撤分布')
        )
        
        # 波动率直方图
        fig.add_trace(
            go.Histogram(x=volatilities, nbinsx=20, name='波动率'),
            row=1, col=1
        )
        
        # 最大回撤直方图
        max_drawdowns = [m['max_drawdown'] for m in risk_metrics.values()]
        fig.add_trace(
            go.Histogram(x=max_drawdowns, nbinsx=20, name='最大回撤'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='风险指标分布',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_recommendation_chart(self, recommendations):
        """创建推荐分布图"""
        categories = ['high_quality', 'high_risk', 'stable', 'avoid']
        counts = [len(recommendations.get(cat, [])) for cat in categories]
        labels = ['优质股', '高风险股', '稳定股', '避免投资']
        
        fig = px.pie(
            values=counts,
            names=labels,
            title='投资建议分布',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def _fig_to_html(self, fig, title):
        """将图表转换为HTML"""
        if not fig.data:
            return ""
        
        chart_html = f"""
        <div class="col-md-6 mb-4">
            <div class="card h-100">
                <div class="card-header">
                    <h5 class="card-title mb-0">{title}</h5>
                </div>
                <div class="card-body">
                    {fig.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </div>
        </div>
        """
        
        return chart_html
    
    def _df_to_html_table(self, df, title):
        """将DataFrame转换为HTML表格"""
        table_html = f"""
        <div class="col-12 mb-4">
            <div class="card h-100">
                <div class="card-header">
                    <h5 class="card-title mb-0">{title}</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        {df.to_html(classes='table table-striped table-hover', index=False)}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return table_html
    
    def _create_history_page(self, results):
        """创建历史数据页面"""
        # 这里可以扩展为显示历史分析结果
        pass
    
    def _get_default_template(self):
        """获取默认HTML模板"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>股票数据分析仪表板</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card-header {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border-radius: 10px 10px 0 0 !important;
        }
        
        .navbar-brand {
            font-weight: bold;
            font-size: 1.5rem;
        }
        
        .metric-card {
            text-align: center;
            padding: 20px;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .metric-label {
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        .last-update {
            font-size: 0.8rem;
            color: #6c757d;
        }
        
        .table th {
            background-color: #f1f3f5;
            border-top: none;
        }
        
        .badge-quality {
            background-color: #28a745;
        }
        
        .badge-risk {
            background-color: #ffc107;
        }
        
        .badge-stable {
            background-color: #17a2b8;
        }
        
        .badge-avoid {
            background-color: #dc3545;
        }
    </style>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-chart-line me-2"></i>股票智能分析系统
            </a>
            <div class="navbar-text text-light last-update">
                最后更新: {{LAST_UPDATE}}
            </div>
        </div>
    </nav>
    
    <!-- 主要内容 -->
    <div class="container-fluid mt-4">
        <!-- 指标卡片 -->
        <div class="row mb-4" id="metrics-section">
            <!-- METRICS_PLACEHOLDER -->
        </div>
        
        <!-- 图表区域 -->
        <div class="row mb-4" id="charts-section">
            <!-- CHARTS_PLACEHOLDER -->
        </div>
        
        <!-- 数据表格 -->
        <div class="row mb-4" id="tables-section">
            <!-- TABLES_PLACEHOLDER -->
        </div>
        
        <!-- 说明区域 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">分析说明</h5>
                    </div>
                    <div class="card-body">
                        <p>本分析基于以下维度：</p>
                        <ul>
                            <li><strong>聚类分析</strong>：基于市值、波动率、技术指标对股票进行分组</li>
                            <li><strong>板块关联</strong>：分析不同板块之间的相关性和因果关系</li>
                            <li><strong>风险度量</strong>：计算VaR、最大回撤、夏普比率等风险指标</li>
                            <li><strong>投资建议</strong>：基于综合分析结果提供投资建议</li>
                        </ul>
                        <p class="mb-0"><small class="text-muted">数据更新频率：每日收盘后 | 分析周期：最近250个交易日</small></p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 页脚 -->
    <footer class="bg-dark text-white py-4 mt-4">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5>股票数据分析系统</h5>
                    <p class="mb-0">基于机器学习的智能化股票分析与推荐系统</p>
                </div>
                <div class="col-md-6 text-md-end">
                    <p class="mb-0">
                        <i class="fas fa-sync-alt me-1"></i>每日自动更新
                        <span class="mx-2">|</span>
                        <i class="fas fa-database me-1"></i>实时数据分析
                    </p>
                </div>
            </div>
        </div>
    </footer>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        // 自动刷新页面（每10分钟）
        setTimeout(function() {
            location.reload();
        }, 600000);
        
        // 图表响应式调整
        window.addEventListener('resize', function() {
            Plotly.Plots.resize(document.getElementById('charts-section'));
        });
        
        // 页面加载动画
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.card');
            cards.forEach((card, index) => {
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 100);
            });
        });
    </script>
</body>
</html>"""
