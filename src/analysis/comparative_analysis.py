"""Comparative analysis for multi-period and multi-site diversity comparison."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import jaccard_score
import warnings

from ..utils.logger import LoggerMixin
from ..utils.data_structures import DiversityMetrics


@dataclass
class ComparisonConfig:
    """Configuration for comparative analysis."""
    
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000
    min_sample_size: int = 5
    correlation_method: str = "spearman"  # pearson, spearman, kendall
    beta_diversity_method: str = "bray_curtis"  # bray_curtis, jaccard, sorensen


class ComparativeAnalyzer(LoggerMixin):
    """Advanced comparative analysis for diversity metrics."""
    
    def __init__(self, config: Optional[ComparisonConfig] = None):
        """Initialize comparative analyzer.
        
        Args:
            config: Configuration for comparative analysis
        """
        self.config = config or ComparisonConfig()
    
    def compare_temporal_diversity(
        self, 
        daily_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare diversity metrics across time periods.
        
        Args:
            daily_summaries: List of daily diversity summaries
            
        Returns:
            Dictionary with temporal comparison results
        """
        if len(daily_summaries) < 2:
            return self._empty_temporal_comparison()
        
        dates = []
        metrics_data: Dict[str, List[Any]] = {}
        
        for summary in daily_summaries:
            if 'date' not in summary or 'diversity_metrics' not in summary:
                continue
                
            dates.append(summary['date'])
            
            for metric_name, value in summary['diversity_metrics'].items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(value)
        
        if not metrics_data:
            return self._empty_temporal_comparison()
        
        df = pd.DataFrame(metrics_data, index=dates)
        df.index = pd.to_datetime(df.index)
        
        results = {
            'period_summary': {
                'start_date': str(df.index.min().date()),
                'end_date': str(df.index.max().date()),
                'total_days': len(df),
                'metrics_analyzed': list(metrics_data.keys())
            },
            'trend_analysis': {},
            'seasonal_patterns': {},
            'statistical_tests': {},
            'change_points': {}
        }
        
        for metric in metrics_data.keys():
            if len(metrics_data[metric]) >= self.config.min_sample_size:
                trend_results = self._analyze_trend(df[metric])
                results['trend_analysis'][metric] = trend_results
        
        if len(df) >= 30:  # At least 30 days for seasonal analysis
            for metric in metrics_data.keys():
                seasonal_results = self._analyze_seasonal_patterns(df[metric])
                results['seasonal_patterns'][metric] = seasonal_results
        
        results['statistical_tests'] = self._perform_temporal_tests(df)
        
        results['change_points'] = self._detect_change_points(df)
        
        return results
    
    def compare_beta_diversity(
        self, 
        site_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate beta diversity (between-site diversity) metrics.
        
        Args:
            site_data: List of site-specific diversity data
            
        Returns:
            Dictionary with beta diversity results
        """
        if len(site_data) < 2:
            return self._empty_beta_diversity()
        
        site_species = {}
        site_names = []
        
        for i, site in enumerate(site_data):
            site_name = site.get('site_name', f'Site_{i+1}')
            site_names.append(site_name)
            
            if 'species_counts' in site:
                site_species[site_name] = site['species_counts']
            elif 'top_species' in site:
                species_counts = {}
                for species_info in site['top_species']:
                    species_name = species_info.get('species_name', '')
                    count = species_info.get('count', 1)
                    species_counts[species_name] = count
                site_species[site_name] = species_counts
        
        if len(site_species) < 2:
            return self._empty_beta_diversity()
        
        all_species_set = set()
        for species_dict in site_species.values():
            all_species_set.update(species_dict.keys())
        
        all_species = sorted(list(all_species_set))
        abundance_matrix_list = []
        
        for site_name in site_names:
            if site_name in site_species:
                abundances = [
                    site_species[site_name].get(species, 0) 
                    for species in all_species
                ]
                abundance_matrix_list.append(abundances)
            else:
                abundance_matrix_list.append([0] * len(all_species))
        
        abundance_matrix = np.array(abundance_matrix_list, dtype=np.float64)
        
        results: Dict[str, Any] = {
            'sites': site_names,
            'total_species': len(all_species),
            'species_list': all_species,
            'beta_diversity_indices': {},
            'similarity_matrix': {},
            'cluster_analysis': {},
            'species_turnover': {}
        }
        
        if self.config.beta_diversity_method == "bray_curtis":
            bray_curtis_dist = self._calculate_bray_curtis(abundance_matrix)
            results['beta_diversity_indices']['bray_curtis'] = bray_curtis_dist
            results['similarity_matrix']['bray_curtis'] = (1 - bray_curtis_dist).tolist()
        
        jaccard_sim = self._calculate_jaccard_similarity(abundance_matrix)
        results['beta_diversity_indices']['jaccard'] = (1 - jaccard_sim).tolist()
        results['similarity_matrix']['jaccard'] = jaccard_sim.tolist()
        
        sorensen_sim = self._calculate_sorensen_similarity(abundance_matrix)
        results['beta_diversity_indices']['sorensen'] = (1 - sorensen_sim).tolist()
        results['similarity_matrix']['sorensen'] = sorensen_sim.tolist()
        
        results['species_turnover'] = self._analyze_species_turnover(site_species, site_names)
        
        return results
    
    def analyze_species_correlations(
        self, 
        daily_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze correlations and co-occurrence patterns between species.
        
        Args:
            daily_summaries: List of daily diversity summaries
            
        Returns:
            Dictionary with species correlation results
        """
        if len(daily_summaries) < self.config.min_sample_size:
            return self._empty_correlation_analysis()
        
        species_data: Dict[str, List[int]] = {}
        dates = []
        
        for summary in daily_summaries:
            if 'date' not in summary or 'top_species' not in summary:
                continue
                
            dates.append(summary['date'])
            
            day_species = {}
            for species_info in summary['top_species']:
                species_name = species_info.get('species_name', '')
                count = species_info.get('count', 0)
                day_species[species_name] = count
            
            for species in day_species:
                if species not in species_data:
                    species_data[species] = []
                species_data[species].append(day_species[species])
            
            for species in species_data:
                if species not in day_species:
                    species_data[species].append(0)
        
        if len(species_data) < 2:
            return self._empty_correlation_analysis()
        
        species_names = list(species_data.keys())
        abundance_matrix = np.array([species_data[species] for species in species_names])
        
        correlation_results = self._calculate_species_correlations(
            abundance_matrix, 
            species_names
        )
        
        cooccurrence_results = self._analyze_cooccurrence_patterns(
            abundance_matrix, 
            species_names
        )
        
        network_results = self._analyze_species_network(
            correlation_results['correlation_matrix'],
            species_names
        )
        
        return {
            'species_analyzed': species_names,
            'total_species': len(species_names),
            'observation_days': len(dates),
            'correlation_analysis': correlation_results,
            'cooccurrence_patterns': cooccurrence_results,
            'network_analysis': network_results,
            'significant_associations': self._find_significant_associations(
                correlation_results, 
                cooccurrence_results
            )
        }
    
    def perform_statistical_tests(
        self, 
        group1_data: List[Dict[str, Any]], 
        group2_data: List[Dict[str, Any]],
        test_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform statistical tests comparing two groups of diversity data.
        
        Args:
            group1_data: First group of diversity summaries
            group2_data: Second group of diversity summaries
            test_metrics: List of metrics to test (default: all available)
            
        Returns:
            Dictionary with statistical test results
        """
        if not group1_data or not group2_data:
            return self._empty_statistical_tests()
        
        group1_metrics = self._extract_metric_values(group1_data)
        group2_metrics = self._extract_metric_values(group2_data)
        
        if not group1_metrics or not group2_metrics:
            return self._empty_statistical_tests()
        
        if test_metrics is None:
            test_metrics = list(set(group1_metrics.keys()) & set(group2_metrics.keys()))
        
        results: Dict[str, Any] = {
            'group1_summary': {
                'sample_size': len(group1_data),
                'metrics': list(group1_metrics.keys())
            },
            'group2_summary': {
                'sample_size': len(group2_data),
                'metrics': list(group2_metrics.keys())
            },
            'statistical_tests': {},
            'effect_sizes': {},
            'summary': {
                'significant_differences': [],
                'non_significant': [],
                'total_tests': len(test_metrics)
            }
        }
        
        for metric in test_metrics:
            if metric in group1_metrics and metric in group2_metrics:
                test_result = self._perform_metric_comparison(
                    group1_metrics[metric],
                    group2_metrics[metric],
                    metric
                )
                results['statistical_tests'][metric] = test_result
                
                effect_size = self._calculate_effect_size(
                    group1_metrics[metric],
                    group2_metrics[metric]
                )
                results['effect_sizes'][metric] = effect_size
                
                if 't_test' in test_result:
                    p_value = test_result['t_test']['p_value']
                    if p_value < self.config.significance_level:
                        results['summary']['significant_differences'].append(metric)
                    else:
                        results['summary']['non_significant'].append(metric)
        
        return results
    
    def _analyze_trend(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series data."""
        x = np.arange(len(time_series))
        y = time_series.values
        
        mask = ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return {'trend': 'insufficient_data'}
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        mk_result = self._mann_kendall_test(y_clean)
        
        return {
            'linear_trend': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'standard_error': float(std_err)
            },
            'mann_kendall': mk_result,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'trend_strength': abs(slope),
            'is_significant': p_value < self.config.significance_level
        }
    
    def _analyze_seasonal_patterns(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal patterns in time series data."""
        day_of_year = time_series.index.dayofyear
        values = time_series.values
        
        mask = ~np.isnan(values)
        day_of_year_clean = day_of_year[mask]
        values_clean = values[mask]
        
        if len(values_clean) < 10:
            return {'seasonal_pattern': 'insufficient_data'}
        
        monthly_data: Dict[int, List[float]] = {}
        for doy, value in zip(day_of_year_clean, values_clean):
            month = datetime.strptime(f'2000 {doy}', '%Y %j').month
            if month not in monthly_data:
                monthly_data[month] = []
            monthly_data[month].append(value)
        
        monthly_means = {month: np.mean(values) for month, values in monthly_data.items()}
        monthly_stds = {month: np.std(values) for month, values in monthly_data.items()}
        
        overall_mean = np.mean(values_clean)
        seasonal_cv = np.std(list(monthly_means.values())) / overall_mean if overall_mean > 0 else 0
        
        return {
            'monthly_patterns': {
                'means': monthly_means,
                'standard_deviations': monthly_stds
            },
            'seasonal_variation_coefficient': float(seasonal_cv),
            'peak_month': int(max(monthly_means.keys(), key=lambda x: float(monthly_means[x]))) if monthly_means else 1,
            'low_month': int(min(monthly_means.keys(), key=lambda x: float(monthly_means[x]))) if monthly_means else 1,
            'seasonal_amplitude': float(max(float(v) for v in monthly_means.values()) - min(float(v) for v in monthly_means.values())) if monthly_means else 0.0
        }
    
    def _perform_temporal_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests for temporal changes."""
        results = {}
        
        for metric in df.columns:
            values = df[metric].dropna().values
            
            if len(values) < self.config.min_sample_size:
                continue
            
            mid_point = len(values) // 2
            first_half = values[:mid_point]
            second_half = values[mid_point:]
            
            t_stat, t_p_value = stats.ttest_ind(first_half, second_half)
            
            u_stat, u_p_value = stats.mannwhitneyu(first_half, second_half, alternative='two-sided')
            
            results[metric] = {
                'period_comparison': {
                    'first_half_mean': float(np.mean(first_half)),
                    'second_half_mean': float(np.mean(second_half)),
                    'mean_difference': float(np.mean(second_half) - np.mean(first_half))
                },
                't_test': {
                    'statistic': float(t_stat),
                    'p_value': float(t_p_value),
                    'significant': t_p_value < self.config.significance_level
                },
                'mann_whitney_u': {
                    'statistic': float(u_stat),
                    'p_value': float(u_p_value),
                    'significant': u_p_value < self.config.significance_level
                }
            }
        
        return results
    
    def _detect_change_points(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect change points in time series data."""
        results = {}
        
        for metric in df.columns:
            values = df[metric].dropna().values
            
            if len(values) < 10:
                continue
            
            change_points = self._cumsum_change_detection(values)
            
            results[metric] = {
                'detected_change_points': change_points,
                'number_of_changes': len(change_points),
                'most_significant_change': max(change_points, key=lambda x: x['magnitude']) if change_points else None
            }
        
        return results
    
    def _calculate_bray_curtis(self, abundance_matrix: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate Bray-Curtis dissimilarity matrix."""
        n_sites = abundance_matrix.shape[0]
        dissimilarity_matrix = np.zeros((n_sites, n_sites))
        
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                numerator = np.sum(np.abs(abundance_matrix[i] - abundance_matrix[j]))
                denominator = np.sum(abundance_matrix[i] + abundance_matrix[j])
                
                if denominator > 0:
                    dissimilarity = numerator / denominator
                else:
                    dissimilarity = 0.0
                
                dissimilarity_matrix[i, j] = dissimilarity
                dissimilarity_matrix[j, i] = dissimilarity
        
        return dissimilarity_matrix
    
    def _calculate_jaccard_similarity(self, abundance_matrix: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate Jaccard similarity matrix."""
        presence_matrix = (abundance_matrix > 0).astype(int)
        
        n_sites = presence_matrix.shape[0]
        similarity_matrix = np.zeros((n_sites, n_sites))
        
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    intersection = np.sum(presence_matrix[i] & presence_matrix[j])
                    union = np.sum(presence_matrix[i] | presence_matrix[j])
                    
                    if union > 0:
                        similarity = intersection / union
                    else:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _calculate_sorensen_similarity(self, abundance_matrix: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate Sorensen similarity matrix."""
        presence_matrix = (abundance_matrix > 0).astype(int)
        
        n_sites = presence_matrix.shape[0]
        similarity_matrix = np.zeros((n_sites, n_sites))
        
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    intersection = np.sum(presence_matrix[i] & presence_matrix[j])
                    sum_species = np.sum(presence_matrix[i]) + np.sum(presence_matrix[j])
                    
                    if sum_species > 0:
                        similarity = (2 * intersection) / sum_species
                    else:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _analyze_species_turnover(
        self, 
        site_species: Dict[str, Dict[str, int]], 
        site_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze species turnover between sites."""
        turnover_results: Dict[str, Any] = {
            'pairwise_turnover': {},
            'overall_turnover': {},
            'unique_species_per_site': {},
            'shared_species': {}
        }
        
        for site_name in site_names:
            if site_name in site_species:
                turnover_results['unique_species_per_site'][site_name] = len(site_species[site_name])
        
        for i, site1 in enumerate(site_names):
            for j, site2 in enumerate(site_names[i+1:], i+1):
                if site1 in site_species and site2 in site_species:
                    species1 = set(site_species[site1].keys())
                    species2 = set(site_species[site2].keys())
                    
                    shared = len(species1 & species2)
                    total_unique = len(species1 | species2)
                    turnover = (total_unique - shared) / total_unique if total_unique > 0 else 0
                    
                    pair_key = f"{site1}_vs_{site2}"
                    turnover_results['pairwise_turnover'][pair_key] = {
                        'turnover_rate': float(turnover),
                        'shared_species': shared,
                        'total_species': total_unique,
                        'site1_unique': len(species1 - species2),
                        'site2_unique': len(species2 - species1)
                    }
        
        return turnover_results
    
    def _calculate_species_correlations(
        self, 
        abundance_matrix: np.ndarray[Any, np.dtype[Any]], 
        species_names: List[str]
    ) -> Dict[str, Any]:
        """Calculate species correlation matrix."""
        n_species = len(species_names)
        correlation_matrix = np.zeros((n_species, n_species))
        p_value_matrix = np.zeros((n_species, n_species))
        
        for i in range(n_species):
            for j in range(n_species):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    p_value_matrix[i, j] = 0.0
                else:
                    if self.config.correlation_method == "pearson":
                        corr, p_val = stats.pearsonr(abundance_matrix[i], abundance_matrix[j])
                    elif self.config.correlation_method == "spearman":
                        corr, p_val = stats.spearmanr(abundance_matrix[i], abundance_matrix[j])
                    elif self.config.correlation_method == "kendall":
                        corr, p_val = stats.kendalltau(abundance_matrix[i], abundance_matrix[j])
                    else:
                        corr, p_val = stats.spearmanr(abundance_matrix[i], abundance_matrix[j])
                    
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    p_value_matrix[i, j] = p_val if not np.isnan(p_val) else 1.0
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'p_value_matrix': p_value_matrix.tolist(),
            'species_names': species_names,
            'method': self.config.correlation_method,
            'significant_correlations': self._find_significant_correlations(
                correlation_matrix, 
                p_value_matrix, 
                species_names
            )
        }
    
    def _analyze_cooccurrence_patterns(
        self, 
        abundance_matrix: np.ndarray[Any, np.dtype[Any]], 
        species_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze species co-occurrence patterns."""
        presence_matrix = (abundance_matrix > 0).astype(int)
        
        cooccurrence_results: Dict[str, Any] = {
            'cooccurrence_matrix': {},
            'association_strength': {},
            'significant_associations': []
        }
        
        n_species = len(species_names)
        n_observations = presence_matrix.shape[1]
        
        for i in range(n_species):
            for j in range(i+1, n_species):
                species1 = species_names[i]
                species2 = species_names[j]
                
                both_present = np.sum(presence_matrix[i] & presence_matrix[j])
                species1_only = np.sum(presence_matrix[i] & ~presence_matrix[j])
                species2_only = np.sum(~presence_matrix[i] & presence_matrix[j])
                neither_present = np.sum(~presence_matrix[i] & ~presence_matrix[j])
                
                contingency_table = np.array([
                    [both_present, species1_only],
                    [species2_only, neither_present]
                ])
                
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    
                    phi = self._calculate_phi_coefficient(contingency_table)
                    
                    pair_key = f"{species1}_x_{species2}"
                    cooccurrence_results['cooccurrence_matrix'][pair_key] = {
                        'both_present': int(both_present),
                        'species1_only': int(species1_only),
                        'species2_only': int(species2_only),
                        'neither_present': int(neither_present),
                        'total_observations': int(n_observations)
                    }
                    
                    cooccurrence_results['association_strength'][pair_key] = {
                        'phi_coefficient': float(phi),
                        'chi_square': float(chi2),
                        'p_value': float(p_value),
                        'is_significant': p_value < self.config.significance_level,
                        'association_type': 'positive' if phi > 0 else 'negative' if phi < 0 else 'none'
                    }
                    
                    if p_value < self.config.significance_level:
                        cooccurrence_results['significant_associations'].append({
                            'species_pair': pair_key,
                            'association_strength': float(phi),
                            'p_value': float(p_value),
                            'type': 'positive' if phi > 0 else 'negative'
                        })
                
                except ValueError:
                    continue
        
        return cooccurrence_results
    
    def _analyze_species_network(
        self, 
        correlation_matrix: List[List[float]], 
        species_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze species interaction network."""
        correlation_array = np.array(correlation_matrix)
        
        threshold = 0.5  # Can be made configurable
        
        adjacency_matrix = (np.abs(correlation_array) > threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-connections
        
        network_results = {
            'adjacency_matrix': adjacency_matrix.tolist(),
            'network_metrics': {},
            'node_metrics': {},
            'communities': {}
        }
        
        total_edges = np.sum(adjacency_matrix) // 2  # Undirected graph
        possible_edges = len(species_names) * (len(species_names) - 1) // 2
        density = total_edges / possible_edges if possible_edges > 0 else 0
        
        network_results['network_metrics'] = {
            'total_nodes': len(species_names),
            'total_edges': int(total_edges),
            'density': float(density),
            'average_degree': float(np.mean(np.sum(adjacency_matrix, axis=1)))
        }
        
        for i, species in enumerate(species_names):
            degree = np.sum(adjacency_matrix[i])
            network_results['node_metrics'][species] = {
                'degree': int(degree),
                'degree_centrality': float(degree / (len(species_names) - 1)) if len(species_names) > 1 else 0
            }
        
        return network_results
    
    def _find_significant_associations(
        self, 
        correlation_results: Dict[str, Any], 
        cooccurrence_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find significant species associations."""
        significant_associations = []
        
        if 'significant_correlations' in correlation_results:
            for corr in correlation_results['significant_correlations']:
                significant_associations.append({
                    'type': 'correlation',
                    'species_pair': corr['species_pair'],
                    'strength': corr['correlation'],
                    'p_value': corr['p_value'],
                    'association_type': 'positive' if corr['correlation'] > 0 else 'negative'
                })
        
        if 'significant_associations' in cooccurrence_results:
            for assoc in cooccurrence_results['significant_associations']:
                significant_associations.append({
                    'type': 'cooccurrence',
                    'species_pair': assoc['species_pair'],
                    'strength': assoc['association_strength'],
                    'p_value': assoc['p_value'],
                    'association_type': assoc['type']
                })
        
        return significant_associations
    
    def _extract_metric_values(self, data_list: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract metric values from data list."""
        metrics: Dict[str, List[float]] = {}
        
        for item in data_list:
            if 'diversity_metrics' in item:
                for metric_name, value in item['diversity_metrics'].items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(float(value))
        
        return metrics
    
    def _perform_metric_comparison(
        self, 
        group1_values: List[float], 
        group2_values: List[float], 
        metric_name: str
    ) -> Dict[str, Any]:
        """Perform statistical comparison between two groups for a specific metric."""
        group1_stats = {
            'mean': float(np.mean(group1_values)),
            'std': float(np.std(group1_values)),
            'median': float(np.median(group1_values)),
            'min': float(np.min(group1_values)),
            'max': float(np.max(group1_values))
        }
        
        group2_stats = {
            'mean': float(np.mean(group2_values)),
            'std': float(np.std(group2_values)),
            'median': float(np.median(group2_values)),
            'min': float(np.min(group2_values)),
            'max': float(np.max(group2_values))
        }
        
        t_stat, t_p_value = stats.ttest_ind(group1_values, group2_values)
        
        u_stat, u_p_value = stats.mannwhitneyu(group1_values, group2_values, alternative='two-sided')
        
        levene_stat, levene_p = stats.levene(group1_values, group2_values)
        
        return {
            'metric': metric_name,
            'group1_stats': group1_stats,
            'group2_stats': group2_stats,
            'mean_difference': group2_stats['mean'] - group1_stats['mean'],
            't_test': {
                'statistic': float(t_stat),
                'p_value': float(t_p_value),
                'significant': t_p_value < self.config.significance_level
            },
            'mann_whitney_u': {
                'statistic': float(u_stat),
                'p_value': float(u_p_value),
                'significant': u_p_value < self.config.significance_level
            },
            'levene_equal_variance': {
                'statistic': float(levene_stat),
                'p_value': float(levene_p),
                'equal_variances': levene_p > self.config.significance_level
            },
            'period_comparison': {
                'group1_mean': group1_stats['mean'],
                'group2_mean': group2_stats['mean'],
                'difference': group2_stats['mean'] - group1_stats['mean'],
                'percent_change': ((group2_stats['mean'] - group1_stats['mean']) / group1_stats['mean'] * 100) if group1_stats['mean'] != 0 else 0
            }
        }
    
    def _calculate_effect_size(self, group1_values: List[float], group2_values: List[float]) -> Dict[str, float]:
        """Calculate effect size measures."""
        mean1, mean2 = np.mean(group1_values), np.mean(group2_values)
        std1, std2 = np.std(group1_values, ddof=1), np.std(group2_values, ddof=1)
        n1, n2 = len(group1_values), len(group2_values)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        glass_delta = (mean2 - mean1) / std1 if std1 > 0 else 0
        
        return {
            'cohens_d': float(cohens_d),
            'glass_delta': float(glass_delta),
            'effect_size_interpretation': self._interpret_effect_size(float(abs(cohens_d)))
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _mann_kendall_test(self, data: np.ndarray[Any, np.dtype[Any]]) -> Dict[str, Any]:
        """Perform Mann-Kendall trend test."""
        n = len(data)
        
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            's_statistic': int(s),
            'z_statistic': float(z),
            'p_value': float(p_value),
            'trend': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend',
            'significant': p_value < self.config.significance_level
        }
    
    def _cumsum_change_detection(self, data: np.ndarray[Any, np.dtype[Any]]) -> List[Dict[str, Any]]:
        """Simple change point detection using cumulative sum."""
        n = len(data)
        if n < 10:
            return []
        
        mean_val = np.mean(data)
        cumsum = np.cumsum(data - mean_val)
        
        change_points = []
        threshold = 2 * np.std(data)  # Simple threshold
        
        for i in range(5, n - 5):  # Avoid edges
            if abs(cumsum[i]) > threshold:
                local_window = cumsum[max(0, i-3):min(n, i+4)]
                if abs(cumsum[i]) == max(abs(local_window)):
                    change_points.append({
                        'position': int(i),
                        'magnitude': float(abs(cumsum[i])),
                        'direction': 'increase' if cumsum[i] > 0 else 'decrease'
                    })
        
        filtered_change_points: List[Dict[str, Any]] = []
        for cp in sorted(change_points, key=lambda x: x['magnitude'], reverse=True):
            if not any(abs(cp['position'] - fcp['position']) < 5 for fcp in filtered_change_points):
                filtered_change_points.append(cp)
        
        return filtered_change_points[:5]  # Return top 5 change points
    
    def _find_significant_correlations(
        self, 
        correlation_matrix: np.ndarray[Any, np.dtype[Any]], 
        p_value_matrix: np.ndarray[Any, np.dtype[Any]], 
        species_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Find significant correlations between species."""
        significant_correlations = []
        
        n_species = len(species_names)
        for i in range(n_species):
            for j in range(i + 1, n_species):
                if p_value_matrix[i, j] < self.config.significance_level:
                    significant_correlations.append({
                        'species_pair': f"{species_names[i]}_x_{species_names[j]}",
                        'correlation': float(correlation_matrix[i, j]),
                        'p_value': float(p_value_matrix[i, j]),
                        'species_1': species_names[i],
                        'species_2': species_names[j]
                    })
        
        significant_correlations.sort(key=lambda x: abs(float(x['correlation'])), reverse=True)
        
        return significant_correlations
    
    def _calculate_phi_coefficient(self, contingency_table: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Calculate Phi coefficient for 2x2 contingency table."""
        a, b = contingency_table[0]
        c, d = contingency_table[1]
        
        numerator = (a * d) - (b * c)
        denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def _empty_temporal_comparison(self) -> Dict[str, Any]:
        """Return empty temporal comparison result."""
        return {
            'period_summary': {},
            'trend_analysis': {},
            'seasonal_patterns': {},
            'statistical_tests': {},
            'change_points': {},
            'error': 'insufficient_data'
        }
    
    def _empty_beta_diversity(self) -> Dict[str, Any]:
        """Return empty beta diversity result."""
        return {
            'sites': [],
            'total_species': 0,
            'species_list': [],
            'beta_diversity_indices': {},
            'similarity_matrix': {},
            'cluster_analysis': {},
            'species_turnover': {},
            'error': 'insufficient_data'
        }
    
    def _empty_correlation_analysis(self) -> Dict[str, Any]:
        """Return empty correlation analysis result."""
        return {
            'species_analyzed': [],
            'total_species': 0,
            'observation_days': 0,
            'correlation_analysis': {},
            'cooccurrence_patterns': {},
            'network_analysis': {},
            'significant_associations': [],
            'error': 'insufficient_data'
        }
    
    def _empty_statistical_tests(self) -> Dict[str, Any]:
        """Return empty statistical tests result."""
        return {
            'group1_summary': {},
            'group2_summary': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'summary': {
                'significant_differences': [],
                'non_significant': [],
                'total_tests': 0
            },
            'error': 'insufficient_data'
        }
