#!/usr/bin/env python3
"""Demo script for comparative analysis."""

from pathlib import Path
from src.analysis.comparative_analysis import ComparativeAnalyzer

def main():
    print("🔬 比較分析テスト")
    print("=" * 50)
    
    analyzer = ComparativeAnalyzer()
    
    print("📊 サンプル日次データ生成中...")
    sample_data = analyzer._generate_sample_daily_summaries()
    
    print("📈 時系列多様性比較分析中...")
    temporal_results = analyzer.compare_temporal_diversity(sample_data)
    
    print("🎯 ベータ多様性分析中...")
    beta_results = analyzer.compare_beta_diversity([sample_data[:15], sample_data[15:]])
    
    print("🔗 種相関分析中...")
    correlation_results = analyzer.analyze_species_correlations(sample_data)
    
    print("✅ 比較分析完了!")
    print(f"📁 日次サマリー数: {len(sample_data)}")
    
    print(f"\n📈 時系列分析結果:")
    if 'temporal_trends' in temporal_results:
        trends = temporal_results['temporal_trends']
        print(f"  トレンド指標数: {len(trends)}")
        for metric, trend_data in list(trends.items())[:3]:
            slope = trend_data.get('slope', 'N/A')
            p_value = trend_data.get('p_value', 'N/A')
            print(f"  {metric}: 傾き={slope}, p値={p_value}")
    
    print(f"\n🎯 ベータ多様性結果:")
    if 'beta_diversity' in beta_results:
        beta_div = beta_results['beta_diversity']
        method = beta_div.get('method', 'N/A')
        mean_distance = beta_div.get('mean_distance', 'N/A')
        print(f"  手法: {method}")
        print(f"  平均距離: {mean_distance}")
    
    print(f"\n🔗 種相関結果:")
    if 'species_correlations' in correlation_results:
        correlations = correlation_results['species_correlations']
        if 'significant_pairs' in correlations:
            sig_pairs = len(correlations['significant_pairs'])
            print(f"  有意な相関ペア数: {sig_pairs}")
    
    print("\n🌐 Webサーバーで /comparative-analysis にアクセスして確認してください!")

if __name__ == "__main__":
    main()
