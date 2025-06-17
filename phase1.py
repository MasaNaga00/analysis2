import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CameraPartsAnalyzer:
    def __init__(self, df_parts, df_sales):
        """
        カメラ部品異常検知・可視化システム
        
        Parameters:
        df_parts: 修理データ（date, HEAD, prod_month, Model, Area, parts_no, IF_ID）
        df_sales: 販売データ（date, Model, Area, QT_ALL）
        """
        self.df_parts = df_parts.copy()
        self.df_sales = df_sales.copy()
        
        # データ前処理
        self._preprocess_data()
        
        # 故障率計算
        self.failure_analysis = self._calculate_failure_rates()
        
        # 異常値検知
        self.anomalies = self._detect_anomalies()
    
    def _preprocess_data(self):
        """データの前処理"""
        # 修理データの月別集計用
        self.df_parts['repair_month'] = pd.to_datetime(self.df_parts['date']).dt.to_period('M')
        
        # 販売データの月別期間型変換
        self.df_sales['sales_month'] = pd.to_datetime(self.df_sales['date']).dt.to_period('M')
        
        print("データ前処理完了")
        print(f"修理データ: {len(self.df_parts)} 行")
        print(f"販売データ: {len(self.df_sales)} 行")
    
    def _calculate_failure_rates(self):
        """故障率の計算"""
        
        # 1. 月別・機種別・地域別・部品別の修理件数
        repair_summary = self.df_parts.groupby([
            'repair_month', 'Model', 'Area', 'parts_no'
        ]).agg({
            'IF_ID': 'nunique',  # ユニーク修理件数
            'parts_no': 'count'   # 部品使用数
        }).rename(columns={'IF_ID': 'repair_count', 'parts_no': 'parts_usage'}).reset_index()
        
        # 2. 月別・機種別・地域別の累積販売台数計算
        sales_cumsum = self.df_sales.groupby(['Model', 'Area']).apply(
            lambda x: x.set_index('sales_month')['QT_ALL'].cumsum()
        ).reset_index()
        sales_cumsum.columns = ['Model', 'Area', 'repair_month', 'cumulative_sales']
        
        # 3. 修理データと販売データの結合
        failure_data = repair_summary.merge(
            sales_cumsum, 
            on=['repair_month', 'Model', 'Area'], 
            how='left'
        )
        
        # 4. 故障率計算（千台あたり）
        failure_data['failure_rate_per_1000'] = (
            failure_data['repair_count'] / failure_data['cumulative_sales'] * 1000
        )
        
        # 5. 部品使用率計算（千台あたり）
        failure_data['parts_usage_rate_per_1000'] = (
            failure_data['parts_usage'] / failure_data['cumulative_sales'] * 1000
        )
        
        return failure_data
    
    def _detect_anomalies(self):
        """異常値検知"""
        anomalies = {}
        
        # 部品別異常検知
        for parts_no in self.failure_analysis['parts_no'].unique():
            parts_data = self.failure_analysis[
                self.failure_analysis['parts_no'] == parts_no
            ]['failure_rate_per_1000'].dropna()
            
            if len(parts_data) > 0:
                # 3σ法による異常検知
                mean_rate = parts_data.mean()
                std_rate = parts_data.std()
                upper_limit_3sigma = mean_rate + 3 * std_rate
                
                # IQR法による異常検知
                Q1 = parts_data.quantile(0.25)
                Q3 = parts_data.quantile(0.75)
                IQR = Q3 - Q1
                upper_limit_iqr = Q3 + 1.5 * IQR
                
                anomalies[parts_no] = {
                    'mean': mean_rate,
                    'std': std_rate,
                    'upper_3sigma': upper_limit_3sigma,
                    'upper_iqr': upper_limit_iqr,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR
                }
        
        return anomalies
    
    def get_anomaly_summary(self, method='3sigma'):
        """異常値サマリーの取得"""
        anomaly_list = []
        
        for _, row in self.failure_analysis.iterrows():
            parts_no = row['parts_no']
            failure_rate = row['failure_rate_per_1000']
            
            if parts_no in self.anomalies and not pd.isna(failure_rate):
                threshold = (self.anomalies[parts_no]['upper_3sigma'] if method == '3sigma' 
                            else self.anomalies[parts_no]['upper_iqr'])
                
                if failure_rate > threshold:
                    anomaly_list.append({
                        'repair_month': row['repair_month'],
                        'Model': row['Model'],
                        'Area': row['Area'],
                        'parts_no': parts_no,
                        'failure_rate': failure_rate,
                        'threshold': threshold,
                        'severity': failure_rate / threshold if threshold > 0 else 0,
                        'repair_count': row['repair_count'],
                        'cumulative_sales': row['cumulative_sales']
                    })
        
        return pd.DataFrame(anomaly_list).sort_values('severity', ascending=False)
    
    def plot_failure_rate_distribution(self, parts_filter=None, model_filter=None, area_filter=None):
        """故障率分布の可視化"""
        
        # データフィルタリング
        data = self.failure_analysis.copy()
        title_parts = []
        
        if parts_filter:
            data = data[data['parts_no'].isin(parts_filter)]
            title_parts.append(f"Parts: {', '.join(parts_filter)}")
        
        if model_filter:
            data = data[data['Model'].isin(model_filter)]
            title_parts.append(f"Model: {', '.join(model_filter)}")
            
        if area_filter:
            data = data[data['Area'].isin(area_filter)]
            title_parts.append(f"Area: {', '.join(area_filter)}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Failure Rate Analysis\n{" | ".join(title_parts) if title_parts else "All Data"}', 
                     fontsize=16, fontweight='bold')
        
        # 1. 故障率のヒストグラム
        axes[0, 0].hist(data['failure_rate_per_1000'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Failure Rate Distribution (per 1000 units)')
        axes[0, 0].set_xlabel('Failure Rate per 1000 units')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 部品別故障率ボックスプロット
        if len(data['parts_no'].unique()) <= 20:  # 部品数が多すぎない場合のみ
            data.boxplot(column='failure_rate_per_1000', by='parts_no', ax=axes[0, 1])
            axes[0, 1].set_title('Failure Rate by Parts')
            axes[0, 1].set_xlabel('Parts Number')
            axes[0, 1].set_ylabel('Failure Rate per 1000 units')
            plt.setp(axes[0, 1].get_xticklabels(), rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'Too many parts to display\n(>20 parts)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Parts Distribution')
        
        # 3. 機種別故障率
        model_stats = data.groupby('Model')['failure_rate_per_1000'].agg(['mean', 'std']).reset_index()
        axes[1, 0].bar(model_stats['Model'], model_stats['mean'], 
                       yerr=model_stats['std'], capsize=5, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Average Failure Rate by Model')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Failure Rate per 1000 units')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 地域別故障率
        area_stats = data.groupby('Area')['failure_rate_per_1000'].agg(['mean', 'std']).reset_index()
        axes[1, 1].bar(area_stats['Area'], area_stats['mean'], 
                       yerr=area_stats['std'], capsize=5, alpha=0.7, color='lightgreen')
        axes[1, 1].set_title('Average Failure Rate by Area')
        axes[1, 1].set_xlabel('Area')
        axes[1, 1].set_ylabel('Failure Rate per 1000 units')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_heatmap(self, method='3sigma'):
        """異常値ヒートマップの可視化"""
        
        # 異常値データの取得
        anomaly_df = self.get_anomaly_summary(method)
        
        if len(anomaly_df) == 0:
            print("No anomalies detected.")
            return
        
        # ピボットテーブル作成（機種×部品の異常度）
        pivot_data = anomaly_df.pivot_table(
            values='severity', 
            index='parts_no', 
            columns='Model', 
            aggfunc='max',
            fill_value=0
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 異常度ヒートマップ
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='Reds', 
                   ax=axes[0], cbar_kws={'label': 'Anomaly Severity'})
        axes[0].set_title(f'Anomaly Severity Heatmap ({method.upper()} method)')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Parts Number')
        
        # 2. 異常値の散布図（故障率 vs 閾値）
        axes[1].scatter(anomaly_df['threshold'], anomaly_df['failure_rate'], 
                       alpha=0.6, s=60, c=anomaly_df['severity'], cmap='Reds')
        axes[1].plot([0, anomaly_df['threshold'].max()], [0, anomaly_df['threshold'].max()], 
                    'k--', alpha=0.5, label='Threshold line')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Actual Failure Rate')
        axes[1].set_title('Anomalies: Actual vs Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.colorbar(axes[1].collections[0], ax=axes[1], label='Severity')
        plt.tight_layout()
        plt.show()
        
        return anomaly_df
    
    def plot_time_series(self, parts_filter=None, model_filter=None, area_filter=None):
        """時系列での故障率推移"""
        
        # データフィルタリング
        data = self.failure_analysis.copy()
        
        if parts_filter:
            data = data[data['parts_no'].isin(parts_filter)]
        if model_filter:
            data = data[data['Model'].isin(model_filter)]
        if area_filter:
            data = data[data['Area'].isin(area_filter)]
        
        # 月別集計
        monthly_data = data.groupby('repair_month').agg({
            'failure_rate_per_1000': 'mean',
            'repair_count': 'sum',
            'parts_usage': 'sum'
        }).reset_index()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. 故障率の時系列
        axes[0].plot(monthly_data['repair_month'].astype(str), 
                    monthly_data['failure_rate_per_1000'], 
                    marker='o', linewidth=2, markersize=4)
        axes[0].set_title('Monthly Failure Rate Trend')
        axes[0].set_ylabel('Failure Rate per 1000 units')
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].get_xticklabels(), rotation=45)
        
        # 2. 修理件数の時系列
        axes[1].bar(monthly_data['repair_month'].astype(str), 
                   monthly_data['repair_count'], alpha=0.7, color='orange')
        axes[1].set_title('Monthly Repair Count')
        axes[1].set_ylabel('Repair Count')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].get_xticklabels(), rotation=45)
        
        # 3. 部品使用数の時系列
        axes[2].bar(monthly_data['repair_month'].astype(str), 
                   monthly_data['parts_usage'], alpha=0.7, color='green')
        axes[2].set_title('Monthly Parts Usage')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('Parts Usage Count')
        axes[2].grid(True, alpha=0.3)
        plt.setp(axes[2].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, method='3sigma', top_n=10):
        """異常検知レポート生成"""
        
        print("="*60)
        print("カメラ部品異常検知レポート")
        print("="*60)
        
        # 基本統計
        total_models = self.df_parts['Model'].nunique()
        total_areas = self.df_parts['Area'].nunique()
        total_parts = self.df_parts['parts_no'].nunique()
        total_repairs = self.df_parts['IF_ID'].nunique()
        
        print(f"分析期間: {self.df_parts['date'].min().strftime('%Y-%m-%d')} ～ {self.df_parts['date'].max().strftime('%Y-%m-%d')}")
        print(f"機種数: {total_models}, 地域数: {total_areas}, 部品種類数: {total_parts}")
        print(f"総修理件数: {total_repairs:,}")
        print()
        
        # 異常値検知結果
        anomalies = self.get_anomaly_summary(method)
        
        print(f"【{method.upper()}法による異常検知結果】")
        print(f"検出された異常値: {len(anomalies)} 件")
        print()
        
        if len(anomalies) > 0:
            print(f"上位 {min(top_n, len(anomalies))} 件の重要異常:")
            print("-" * 80)
            
            for i, (_, row) in enumerate(anomalies.head(top_n).iterrows(), 1):
                print(f"{i:2d}. {row['Model']}-{row['Area']}-{row['parts_no']} "
                      f"({row['repair_month']})")
                print(f"    故障率: {row['failure_rate']:.2f} (閾値: {row['threshold']:.2f})")
                print(f"    重要度: {row['severity']:.2f}倍, 修理件数: {row['repair_count']}")
                print()
        
        return anomalies


# 使用例とテスト用データ生成
def generate_sample_data():
    """テスト用サンプルデータ生成"""
    
    np.random.seed(42)
    
    # 修理データ生成
    models = ['M100', 'M150', 'M200', 'M250']
    areas = ['JP', 'CN', 'USA', 'EUR']
    parts = [f'P{i:03d}' for i in range(1, 21)]  # P001-P020
    
    repair_data = []
    start_date = datetime(2023, 1, 1)
    
    for _ in range(5000):  # 5000件の修理記録
        repair_date = start_date + timedelta(days=np.random.randint(0, 850))
        model = np.random.choice(models)
        area = np.random.choice(areas)
        parts_no = np.random.choice(parts)
        
        # 一部の部品に異常な故障率を設定
        if parts_no in ['P001', 'P005', 'P010']:
            # 異常部品は故障確率を高く設定
            if np.random.random() < 0.3:  # 30%の確率で異常データ生成
                parts_no = np.random.choice(['P001', 'P005', 'P010'])
        
        repair_data.append({
            'date': repair_date,
            'HEAD': f'H{repair_date.year}{repair_date.month:02d}',
            'prod_month': pd.Period(f'{repair_date.year}-{repair_date.month}', 'M'),
            'Model': model,
            'Area': area,
            'parts_no': parts_no,
            'IF_ID': f'IF{len(repair_data):06d}'
        })
    
    df_parts = pd.DataFrame(repair_data)
    
    # 販売データ生成
    sales_data = []
    for year in range(2023, 2026):
        for month in range(1, 13):
            if year == 2025 and month > 5:  # 2025年5月まで
                break
            
            date = datetime(year, month, 1)
            for model in models:
                for area in areas:
                    # 機種・地域による販売台数の差を設定
                    base_sales = {'M100': 1000, 'M150': 800, 'M200': 1200, 'M250': 600}
                    area_factor = {'JP': 1.2, 'CN': 1.5, 'USA': 1.0, 'EUR': 0.8}
                    
                    sales = int(base_sales[model] * area_factor[area] * (1 + np.random.normal(0, 0.2)))
                    sales = max(100, sales)  # 最低100台
                    
                    sales_data.append({
                        'date': date,
                        'Model': model,
                        'Area': area,
                        'QT_ALL': sales
                    })
    
    df_sales = pd.DataFrame(sales_data)
    
    return df_parts, df_sales


# 実行例
if __name__ == "__main__":
    # サンプルデータ生成
    df_parts, df_sales = generate_sample_data()
    
    print("サンプルデータを生成しました。")
    print(f"修理データ: {len(df_parts)} 行")
    print(f"販売データ: {len(df_sales)} 行")
    print()
    
    # 分析システム初期化
    analyzer = CameraPartsAnalyzer(df_parts, df_sales)
    
    # レポート生成
    anomalies = analyzer.generate_report()
    
    # 可視化実行
    print("可視化を実行します...")
    
    # 1. 全体的な故障率分布
    analyzer.plot_failure_rate_distribution()
    
    # 2. 異常値ヒートマップ
    analyzer.plot_anomaly_heatmap()
    
    # 3. 特定部品の時系列分析（異常部品をフィルタ）
    analyzer.plot_time_series(parts_filter=['P001', 'P005', 'P010'])
    
    # 4. 特定機種の分析
    analyzer.plot_failure_rate_distribution(model_filter=['M100', 'M150'])
    