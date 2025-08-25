#!/usr/bin/env python3
"""Demo script for functional diversity analysis."""

from pathlib import Path
from src.analysis.functional_diversity import FunctionalDiversityAnalyzer

def main():
    print("🌿 機能的多様性分析テスト")
    print("=" * 50)
    
    analyzer = FunctionalDiversityAnalyzer()
    
    print("🧬 サンプル形質データベース生成中...")
    species_list = ["Taraxacum officinale", "Plantago major", "Trifolium repens", "Poa annua", "Bellis perennis"]
    sample_traits = analyzer.generate_sample_trait_database(species_list)
    analyzer.load_traits_from_dict(sample_traits)
    
    print("📊 機能的多様性計算中...")
    sample_abundances = {species: 10 - i*2 for i, species in enumerate(species_list)}
    results = analyzer.calculate_functional_diversity(sample_abundances)
    
    print("🔍 形質相関分析中...")
    correlation_results = analyzer.analyze_trait_correlations()
    
    print("🎯 機能群識別中...")
    functional_groups = analyzer.identify_functional_groups()
    
    print("✅ 機能的多様性分析完了!")
    print(f"📁 種数: {len(species_list)}")
    print(f"🧬 形質数: {len(sample_traits[species_list[0]])}")
    
    print(f"\n📊 機能的多様性指標:")
    if 'functional_diversity' in results:
        fd_metrics = results['functional_diversity']
        for metric, value in fd_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    print(f"\n🔍 形質相関:")
    if correlation_results and 'correlation_matrix' in correlation_results:
        matrix = correlation_results['correlation_matrix']
        print(f"  相関行列サイズ: {len(matrix)}x{len(matrix[0])}")
        if 'significant_correlations' in correlation_results:
            sig_corr = len(correlation_results['significant_correlations'])
            print(f"  有意な相関数: {sig_corr}")
    
    print(f"\n🎯 機能群:")
    if functional_groups and 'group_assignments' in functional_groups:
        assignments = functional_groups['group_assignments']
        unique_groups = len(set(assignments.values()))
        print(f"  機能群数: {unique_groups}")
        for species, group in assignments.items():
            print(f"  {species}: グループ {group}")
    
    print("\n🌐 Webサーバーで /functional-diversity にアクセスして確認してください!")

if __name__ == "__main__":
    main()
