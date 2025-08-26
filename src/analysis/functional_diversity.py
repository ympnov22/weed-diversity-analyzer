"""Functional diversity analysis for ecological trait-based metrics."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..utils.logger import LoggerMixin


@dataclass
class FunctionalTraits:
    """Functional traits for species."""
    
    species_name: str
    height_cm: Optional[float] = None
    leaf_area_cm2: Optional[float] = None
    seed_mass_mg: Optional[float] = None
    growth_form: Optional[str] = None  # annual, perennial, woody
    photosynthesis_type: Optional[str] = None  # C3, C4, CAM
    nitrogen_fixation: Optional[bool] = None
    dispersal_mode: Optional[str] = None  # wind, animal, water, ballistic
    flowering_start_month: Optional[int] = None
    flowering_duration_months: Optional[int] = None
    root_depth_cm: Optional[float] = None
    specific_leaf_area: Optional[float] = None  # cm2/g
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert traits to dictionary."""
        return {
            'species_name': self.species_name,
            'height_cm': self.height_cm,
            'leaf_area_cm2': self.leaf_area_cm2,
            'seed_mass_mg': self.seed_mass_mg,
            'growth_form': self.growth_form,
            'photosynthesis_type': self.photosynthesis_type,
            'nitrogen_fixation': self.nitrogen_fixation,
            'dispersal_mode': self.dispersal_mode,
            'flowering_start_month': self.flowering_start_month,
            'flowering_duration_months': self.flowering_duration_months,
            'root_depth_cm': self.root_depth_cm,
            'specific_leaf_area': self.specific_leaf_area
        }


@dataclass
class FunctionalDiversityConfig:
    """Configuration for functional diversity analysis."""
    
    distance_metric: str = "euclidean"  # euclidean, manhattan, gower
    standardize_traits: bool = True
    handle_missing_traits: str = "mean_imputation"  # mean_imputation, median_imputation, drop
    pca_components: Optional[int] = None  # None for auto-selection
    clustering_method: str = "ward"  # ward, complete, average
    functional_groups_threshold: float = 0.5


class FunctionalDiversityAnalyzer(LoggerMixin):
    """Analyze functional diversity based on species traits."""
    
    def __init__(self, config: Optional[FunctionalDiversityConfig] = None):
        """Initialize functional diversity analyzer.
        
        Args:
            config: Configuration for functional diversity analysis
        """
        self.config = config or FunctionalDiversityConfig()
        self.trait_database: Dict[str, FunctionalTraits] = {}
        self.scaler = StandardScaler()
    
    def add_species_traits(self, traits: FunctionalTraits):
        """Add functional traits for a species.
        
        Args:
            traits: Functional traits for the species
        """
        self.trait_database[traits.species_name] = traits
    
    def load_traits_from_dict(self, traits_dict: Dict[str, Dict[str, Any]]):
        """Load traits from dictionary.
        
        Args:
            traits_dict: Dictionary mapping species names to trait dictionaries
        """
        for species_name, trait_data in traits_dict.items():
            traits = FunctionalTraits(species_name=species_name, **trait_data)
            self.add_species_traits(traits)
    
    def calculate_functional_diversity(
        self, 
        species_abundances: Dict[str, float],
        trait_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate functional diversity metrics.
        
        Args:
            species_abundances: Dictionary mapping species names to abundances
            trait_subset: Optional list of traits to use (default: all available)
            
        Returns:
            Dictionary with functional diversity metrics
        """
        if not species_abundances:
            return self._empty_functional_diversity()
        
        species_with_traits = [
            species for species in species_abundances.keys() 
            if species in self.trait_database
        ]
        
        if len(species_with_traits) < 2:
            return self._empty_functional_diversity()
        
        trait_matrix, trait_names = self._create_trait_matrix(
            species_with_traits, 
            trait_subset
        )
        
        if trait_matrix.size == 0:
            return self._empty_functional_diversity()
        
        filtered_abundances = {
            species: species_abundances[species] 
            for species in species_with_traits
        }
        
        results = {
            'species_analyzed': species_with_traits,
            'traits_used': trait_names,
            'trait_matrix_shape': trait_matrix.shape,
            'functional_diversity_indices': {},
            'functional_groups': {},
            'trait_analysis': {},
            'functional_space': {}
        }
        
        fric = self._calculate_functional_richness(trait_matrix)
        results['functional_diversity_indices']['functional_richness'] = fric
        
        feve = self._calculate_functional_evenness(trait_matrix, filtered_abundances)
        results['functional_diversity_indices']['functional_evenness'] = feve
        
        fdiv = self._calculate_functional_divergence(trait_matrix, filtered_abundances)
        results['functional_diversity_indices']['functional_divergence'] = fdiv
        
        fdis = self._calculate_functional_dispersion(trait_matrix, filtered_abundances)
        results['functional_diversity_indices']['functional_dispersion'] = fdis
        
        raos_q = self._calculate_raos_quadratic_entropy(trait_matrix, filtered_abundances)
        results['functional_diversity_indices']['raos_quadratic_entropy'] = raos_q
        
        functional_groups = self._identify_functional_groups(trait_matrix, species_with_traits)
        results['functional_groups'] = functional_groups
        
        trait_analysis = self._analyze_trait_patterns(trait_matrix, trait_names, filtered_abundances)
        results['trait_analysis'] = trait_analysis
        
        functional_space = self._analyze_functional_space(trait_matrix, species_with_traits)
        results['functional_space'] = functional_space
        
        return results
    
    def compare_functional_diversity(
        self, 
        community1_abundances: Dict[str, float],
        community2_abundances: Dict[str, float],
        trait_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare functional diversity between two communities.
        
        Args:
            community1_abundances: Species abundances for community 1
            community2_abundances: Species abundances for community 2
            trait_subset: Optional list of traits to use
            
        Returns:
            Dictionary with functional diversity comparison
        """
        fd1 = self.calculate_functional_diversity(community1_abundances, trait_subset)
        fd2 = self.calculate_functional_diversity(community2_abundances, trait_subset)
        
        if 'error' in fd1 or 'error' in fd2:
            return {'error': 'insufficient_data_for_comparison'}
        
        comparison_results = {
            'community1': fd1,
            'community2': fd2,
            'comparison': {
                'functional_diversity_differences': {},
                'shared_functional_space': {},
                'functional_beta_diversity': {}
            }
        }
        
        for index_name in fd1['functional_diversity_indices']:
            if index_name in fd2['functional_diversity_indices']:
                value1 = fd1['functional_diversity_indices'][index_name]
                value2 = fd2['functional_diversity_indices'][index_name]
                
                comparison_results['comparison']['functional_diversity_differences'][index_name] = {
                    'community1_value': value1,
                    'community2_value': value2,
                    'difference': value2 - value1,
                    'relative_difference': (value2 - value1) / value1 if value1 != 0 else 0
                }
        
        functional_beta = self._calculate_functional_beta_diversity(
            community1_abundances, 
            community2_abundances,
            trait_subset
        )
        comparison_results['comparison']['functional_beta_diversity'] = functional_beta
        
        return comparison_results
    
    def analyze_trait_correlations(
        self, 
        species_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between functional traits.
        
        Args:
            species_list: Optional list of species to analyze (default: all)
            
        Returns:
            Dictionary with trait correlation analysis
        """
        if species_list is None:
            species_list = list(self.trait_database.keys())
        
        trait_matrix, trait_names = self._create_trait_matrix(species_list)
        
        if trait_matrix.size == 0:
            return {'error': 'no_trait_data_available'}
        
        correlation_matrix = np.corrcoef(trait_matrix.T)
        
        significant_correlations = []
        n_traits = len(trait_names)
        
        for i in range(n_traits):
            for j in range(i + 1, n_traits):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > 0.5:  # Threshold for significant correlation
                    significant_correlations.append({
                        'trait1': trait_names[i],
                        'trait2': trait_names[j],
                        'correlation': float(correlation),
                        'strength': 'strong' if abs(correlation) > 0.7 else 'moderate'
                    })
        
        pca_results = self._perform_trait_pca(trait_matrix, trait_names)
        
        return {
            'species_analyzed': species_list,
            'traits_analyzed': trait_names,
            'correlation_matrix': correlation_matrix.tolist(),
            'significant_correlations': significant_correlations,
            'pca_analysis': pca_results,
            'trait_summary_statistics': self._calculate_trait_summary_stats(trait_matrix, trait_names)
        }
    
    def generate_sample_trait_database(self, species_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate sample trait database for testing.
        
        Args:
            species_list: List of species names
            
        Returns:
            Dictionary with sample trait data
        """
        np.random.seed(42)  # For reproducible results
        
        sample_traits = {}
        
        growth_forms = ['annual', 'perennial', 'woody']
        photosynthesis_types = ['C3', 'C4']
        dispersal_modes = ['wind', 'animal', 'water', 'ballistic']
        
        for species in species_list:
            sample_traits[species] = {
                'height_cm': float(np.random.lognormal(3, 0.5)),  # Log-normal distribution
                'leaf_area_cm2': float(np.random.lognormal(2, 0.8)),
                'seed_mass_mg': float(np.random.lognormal(1, 1.2)),
                'growth_form': np.random.choice(growth_forms),
                'photosynthesis_type': np.random.choice(photosynthesis_types),
                'nitrogen_fixation': bool(np.random.choice([True, False], p=[0.2, 0.8])),
                'dispersal_mode': np.random.choice(dispersal_modes),
                'flowering_start_month': int(np.random.randint(3, 8)),  # March to July
                'flowering_duration_months': int(np.random.randint(1, 5)),  # 1-4 months
                'root_depth_cm': float(np.random.lognormal(3.5, 0.6)),
                'specific_leaf_area': float(np.random.lognormal(2.5, 0.4))
            }
        
        return sample_traits
    
    def _create_trait_matrix(
        self, 
        species_list: List[str], 
        trait_subset: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Create trait matrix for analysis."""
        if not species_list:
            return np.array([]), []
        
        all_traits = set()
        for species in species_list:
            if species in self.trait_database:
                trait_dict = self.trait_database[species].to_dict()
                all_traits.update(trait_dict.keys())
        
        all_traits.discard('species_name')  # Remove species name
        
        if trait_subset:
            traits_to_use = [t for t in trait_subset if t in all_traits]
        else:
            traits_to_use = list(all_traits)
        
        if not traits_to_use:
            return np.array([]), []
        
        trait_data = []
        valid_species = []
        
        for species in species_list:
            if species not in self.trait_database:
                continue
            
            trait_dict = self.trait_database[species].to_dict()
            species_traits: List[float] = []
            
            for trait_name in traits_to_use:
                value = trait_dict.get(trait_name)
                
                if trait_name == 'growth_form':
                    if value == 'annual':
                        species_traits.append(0.0)
                    elif value == 'perennial':
                        species_traits.append(1.0)
                    elif value == 'woody':
                        species_traits.append(2.0)
                    else:
                        species_traits.append(np.nan)
                elif trait_name == 'photosynthesis_type':
                    if value == 'C3':
                        species_traits.append(0.0)
                    elif value == 'C4':
                        species_traits.append(1.0)
                    elif value == 'CAM':
                        species_traits.append(2.0)
                    else:
                        species_traits.append(np.nan)
                elif trait_name == 'dispersal_mode':
                    if value == 'wind':
                        species_traits.append(0.0)
                    elif value == 'animal':
                        species_traits.append(1.0)
                    elif value == 'water':
                        species_traits.append(2.0)
                    elif value == 'ballistic':
                        species_traits.append(3.0)
                    else:
                        species_traits.append(np.nan)
                elif trait_name == 'nitrogen_fixation':
                    species_traits.append(1.0 if value else 0.0)
                else:
                    species_traits.append(float(value) if value is not None else np.nan)
            
            trait_data.append(species_traits)
            valid_species.append(species)
        
        if not trait_data:
            return np.array([]), []
        
        trait_matrix = np.array(trait_data)
        
        if self.config.handle_missing_traits == "mean_imputation":
            for col in range(trait_matrix.shape[1]):
                col_data = trait_matrix[:, col]
                mask = ~np.isnan(col_data)
                if np.any(mask):
                    mean_val = np.mean(col_data[mask])
                    trait_matrix[~mask, col] = mean_val
        elif self.config.handle_missing_traits == "median_imputation":
            for col in range(trait_matrix.shape[1]):
                col_data = trait_matrix[:, col]
                mask = ~np.isnan(col_data)
                if np.any(mask):
                    median_val = np.median(col_data[mask])
                    trait_matrix[~mask, col] = median_val
        
        valid_rows = ~np.all(np.isnan(trait_matrix), axis=1)
        valid_cols = ~np.all(np.isnan(trait_matrix), axis=0)
        
        trait_matrix = trait_matrix[valid_rows][:, valid_cols]
        valid_species = [valid_species[i] for i in range(len(valid_species)) if valid_rows[i]]
        traits_to_use = [traits_to_use[i] for i in range(len(traits_to_use)) if valid_cols[i]]
        
        if self.config.standardize_traits and trait_matrix.size > 0:
            trait_matrix = self.scaler.fit_transform(trait_matrix)
        
        return trait_matrix, traits_to_use
    
    def _calculate_functional_richness(self, trait_matrix: np.ndarray) -> float:
        """Calculate functional richness (FRic)."""
        if trait_matrix.shape[0] < 2:
            return 0.0
        
        try:
            from scipy.spatial import ConvexHull
            
            if trait_matrix.shape[1] == 1:
                return float(np.max(trait_matrix) - np.min(trait_matrix))
            elif trait_matrix.shape[1] == 2:
                hull = ConvexHull(trait_matrix)
                return float(hull.volume)
            else:
                hull = ConvexHull(trait_matrix)
                return float(hull.volume)
        except:
            ranges = np.max(trait_matrix, axis=0) - np.min(trait_matrix, axis=0)
            return float(np.prod(ranges))
    
    def _calculate_functional_evenness(
        self, 
        trait_matrix: np.ndarray, 
        abundances: Dict[str, float]
    ) -> float:
        """Calculate functional evenness (FEve)."""
        if trait_matrix.shape[0] < 2:
            return 0.0
        
        distances = pdist(trait_matrix, metric=self.config.distance_metric)
        distance_matrix = squareform(distances)
        
        from scipy.sparse.csgraph import minimum_spanning_tree
        mst = minimum_spanning_tree(distance_matrix).toarray()
        
        species_names = list(abundances.keys())
        total_abundance = sum(abundances.values())
        
        weighted_branch_lengths = []
        for i in range(len(species_names)):
            for j in range(i + 1, len(species_names)):
                if mst[i, j] > 0:
                    weight = (abundances[species_names[i]] + abundances[species_names[j]]) / (2 * total_abundance)
                    weighted_branch_lengths.append(mst[i, j] * weight)
        
        if not weighted_branch_lengths:
            return 0.0
        
        mean_weighted_length = np.mean(weighted_branch_lengths)
        sum_deviations = sum(abs(length - mean_weighted_length) for length in weighted_branch_lengths)
        
        if len(weighted_branch_lengths) > 1:
            evenness = 1 - sum_deviations / (len(weighted_branch_lengths) * mean_weighted_length)
            return max(0.0, float(evenness))
        else:
            return 1.0
    
    def _calculate_functional_divergence(
        self, 
        trait_matrix: np.ndarray, 
        abundances: Dict[str, float]
    ) -> float:
        """Calculate functional divergence (FDiv)."""
        if trait_matrix.shape[0] < 2:
            return 0.0
        
        species_names = list(abundances.keys())
        total_abundance = sum(abundances.values())
        
        abundance_weights = np.array([abundances[species] / total_abundance for species in species_names])
        centroid = np.average(trait_matrix, axis=0, weights=abundance_weights)
        
        distances_from_centroid = np.sqrt(np.sum((trait_matrix - centroid) ** 2, axis=1))
        
        mean_distance = np.average(distances_from_centroid, weights=abundance_weights)
        
        deviations = np.abs(distances_from_centroid - mean_distance)
        weighted_deviations = deviations * abundance_weights
        
        sum_weighted_deviations = np.sum(weighted_deviations)
        sum_distances = np.sum(distances_from_centroid * abundance_weights)
        
        if sum_distances > 0:
            divergence = sum_weighted_deviations / sum_distances
            return float(divergence)
        else:
            return 0.0
    
    def _calculate_functional_dispersion(
        self, 
        trait_matrix: np.ndarray, 
        abundances: Dict[str, float]
    ) -> float:
        """Calculate functional dispersion (FDis)."""
        if trait_matrix.shape[0] < 2:
            return 0.0
        
        species_names = list(abundances.keys())
        total_abundance = sum(abundances.values())
        
        abundance_weights = np.array([abundances[species] / total_abundance for species in species_names])
        centroid = np.average(trait_matrix, axis=0, weights=abundance_weights)
        
        distances_from_centroid = np.sqrt(np.sum((trait_matrix - centroid) ** 2, axis=1))
        
        weighted_mean_distance = np.average(distances_from_centroid, weights=abundance_weights)
        
        return float(weighted_mean_distance)
    
    def _calculate_raos_quadratic_entropy(
        self, 
        trait_matrix: np.ndarray, 
        abundances: Dict[str, float]
    ) -> float:
        """Calculate Rao's quadratic entropy."""
        if trait_matrix.shape[0] < 2:
            return 0.0
        
        distances = pdist(trait_matrix, metric=self.config.distance_metric)
        distance_matrix = squareform(distances)
        
        species_names = list(abundances.keys())
        total_abundance = sum(abundances.values())
        
        raos_q = 0.0
        for i, species_i in enumerate(species_names):
            for j, species_j in enumerate(species_names):
                pi = abundances[species_i] / total_abundance
                pj = abundances[species_j] / total_abundance
                dij = distance_matrix[i, j]
                raos_q += pi * pj * dij
        
        return float(raos_q)
    
    def _identify_functional_groups(
        self, 
        trait_matrix: np.ndarray, 
        species_names: List[str]
    ) -> Dict[str, Any]:
        """Identify functional groups using hierarchical clustering."""
        if trait_matrix.shape[0] < 2:
            return {'functional_groups': {}, 'number_of_groups': 0}
        
        distances = pdist(trait_matrix, metric=self.config.distance_metric)
        linkage_matrix = linkage(distances, method=self.config.clustering_method)
        
        cluster_labels = fcluster(
            linkage_matrix, 
            t=self.config.functional_groups_threshold, 
            criterion='distance'
        )
        
        functional_groups: Dict[str, List[str]] = {}
        for i, species in enumerate(species_names):
            group_id = f"Group_{cluster_labels[i]}"
            if group_id not in functional_groups:
                functional_groups[group_id] = []
            functional_groups[group_id].append(species)
        
        return {
            'functional_groups': functional_groups,
            'number_of_groups': len(functional_groups),
            'cluster_labels': cluster_labels.tolist(),
            'linkage_matrix': linkage_matrix.tolist()
        }
    
    def _analyze_trait_patterns(
        self, 
        trait_matrix: np.ndarray, 
        trait_names: List[str], 
        abundances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze patterns in trait values."""
        if trait_matrix.size == 0:
            return {}
        
        species_names = list(abundances.keys())
        total_abundance = sum(abundances.values())
        abundance_weights = np.array([abundances[species] / total_abundance for species in species_names])
        
        trait_analysis = {}
        
        for i, trait_name in enumerate(trait_names):
            trait_values = trait_matrix[:, i]
            
            trait_analysis[trait_name] = {
                'mean': float(np.mean(trait_values)),
                'std': float(np.std(trait_values)),
                'min': float(np.min(trait_values)),
                'max': float(np.max(trait_values)),
                'range': float(np.max(trait_values) - np.min(trait_values)),
                'weighted_mean': float(np.average(trait_values, weights=abundance_weights)),
                'coefficient_of_variation': float(np.std(trait_values) / np.mean(trait_values)) if np.mean(trait_values) != 0 else 0
            }
        
        return trait_analysis
    
    def _analyze_functional_space(
        self, 
        trait_matrix: np.ndarray, 
        species_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze functional space using PCA."""
        if trait_matrix.shape[0] < 2 or trait_matrix.shape[1] < 2:
            return {'error': 'insufficient_data_for_pca'}
        
        pca = PCA()
        pca_scores = pca.fit_transform(trait_matrix)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.8) + 1
        
        return {
            'pca_scores': pca_scores.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'components_for_80_percent': int(n_components),
            'total_components': len(pca.explained_variance_ratio_),
            'species_names': species_names
        }
    
    def _calculate_functional_beta_diversity(
        self, 
        community1_abundances: Dict[str, float],
        community2_abundances: Dict[str, float],
        trait_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate functional beta diversity between communities."""
        all_species_set = set(community1_abundances.keys()) | set(community2_abundances.keys())
        all_species = [s for s in all_species_set if s in self.trait_database]
        
        if len(all_species) < 2:
            return {'error': 'insufficient_species_with_traits'}
        
        trait_matrix, trait_names = self._create_trait_matrix(all_species, trait_subset)
        
        if trait_matrix.size == 0:
            return {'error': 'no_trait_data'}
        
        abundance1 = np.array([community1_abundances.get(species, 0) for species in all_species])
        abundance2 = np.array([community2_abundances.get(species, 0) for species in all_species])
        
        abundance1 = abundance1 / np.sum(abundance1) if np.sum(abundance1) > 0 else abundance1
        abundance2 = abundance2 / np.sum(abundance2) if np.sum(abundance2) > 0 else abundance2
        
        
        shared_functional_space = self._calculate_shared_functional_space(
            trait_matrix, abundance1, abundance2
        )
        
        community1_space = self._calculate_functional_richness(
            trait_matrix[abundance1 > 0]
        ) if np.any(abundance1 > 0) else 0
        
        community2_space = self._calculate_functional_richness(
            trait_matrix[abundance2 > 0]
        ) if np.any(abundance2 > 0) else 0
        
        total_space = max(community1_space, community2_space)
        
        if total_space > 0:
            functional_turnover = (total_space - shared_functional_space) / total_space
        else:
            functional_turnover = 0
        
        min_space = min(community1_space, community2_space)
        if total_space > 0:
            functional_nestedness = (shared_functional_space - min_space) / total_space
        else:
            functional_nestedness = 0
        
        functional_beta_total = functional_turnover + functional_nestedness
        
        return {
            'functional_beta_total': float(functional_beta_total),
            'functional_turnover': float(functional_turnover),
            'functional_nestedness': float(functional_nestedness),
            'community1_functional_richness': float(community1_space),
            'community2_functional_richness': float(community2_space),
            'shared_functional_space': float(shared_functional_space),
            'species_analyzed': all_species
        }
    
    def _calculate_shared_functional_space(
        self, 
        trait_matrix: np.ndarray, 
        abundance1: np.ndarray, 
        abundance2: np.ndarray
    ) -> float:
        """Calculate shared functional space between two communities."""
        shared_species_mask = (abundance1 > 0) & (abundance2 > 0)
        
        if not np.any(shared_species_mask):
            return 0.0
        
        shared_trait_matrix = trait_matrix[shared_species_mask]
        
        if shared_trait_matrix.shape[0] < 2:
            return 0.0
        
        return self._calculate_functional_richness(shared_trait_matrix)
    
    def _perform_trait_pca(
        self, 
        trait_matrix: np.ndarray, 
        trait_names: List[str]
    ) -> Dict[str, Any]:
        """Perform PCA on trait matrix."""
        if trait_matrix.shape[0] < 2 or trait_matrix.shape[1] < 2:
            return {'error': 'insufficient_data_for_pca'}
        
        pca = PCA()
        pca_scores = pca.fit_transform(trait_matrix)
        
        loadings = pca.components_.T
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'loadings': loadings.tolist(),
            'trait_names': trait_names,
            'n_components': len(pca.explained_variance_ratio_)
        }
    
    def _calculate_trait_summary_stats(
        self, 
        trait_matrix: np.ndarray, 
        trait_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each trait."""
        summary_stats = {}
        
        for i, trait_name in enumerate(trait_names):
            trait_values = trait_matrix[:, i]
            
            summary_stats[trait_name] = {
                'mean': float(np.mean(trait_values)),
                'std': float(np.std(trait_values)),
                'min': float(np.min(trait_values)),
                'max': float(np.max(trait_values)),
                'median': float(np.median(trait_values)),
                'q25': float(np.percentile(trait_values, 25)),
                'q75': float(np.percentile(trait_values, 75)),
                'skewness': float(self._calculate_skewness(trait_values)),
                'kurtosis': float(self._calculate_kurtosis(trait_values))
            }
        
        return summary_stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def _empty_functional_diversity(self) -> Dict[str, Any]:
        """Return empty functional diversity result."""
        return {
            'species_analyzed': [],
            'traits_used': [],
            'trait_matrix_shape': (0, 0),
            'functional_diversity_indices': {},
            'functional_groups': {},
            'trait_analysis': {},
            'functional_space': {},
            'error': 'insufficient_data'
        }
