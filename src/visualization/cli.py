"""Command line interface for visualization server."""

import click
import uvicorn
from pathlib import Path

from .web_server import WebServer
from ..output.output_manager import OutputManager
from ..utils.config import ConfigManager


@click.group()
def cli():
    """Weed Diversity Analyzer Visualization CLI."""
    pass


@cli.command()
@click.option('--host', default='127.0.0.1', help='Server host')
@click.option('--port', default=8000, help='Server port')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--output-dir', type=click.Path(), help='Output directory path')
def serve(host, port, debug, output_dir):
    """Start the visualization web server."""
    try:
        if output_dir:
            output_manager = OutputManager(Path(output_dir))
        else:
            output_manager = OutputManager()
        
        server = WebServer(
            output_manager=output_manager,
            host=host,
            port=port
        )
        
        click.echo(f"Starting visualization server at http://{host}:{port}")
        server.run(debug=debug)
        
    except Exception as e:
        click.echo(f"Failed to start server: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--metric', default='shannon_diversity', help='Diversity metric for calendar')
@click.option('--year', type=int, help='Year to generate calendar for')
@click.option('--output', type=click.Path(), help='Output HTML file path')
def calendar(metric, year, output):
    """Generate GitHub-style calendar visualization."""
    try:
        from .calendar_visualizer_stub import CalendarVisualizer
        
        visualizer = CalendarVisualizer()
        sample_data = visualizer.generate_sample_data(year=year)
        
        output_path = Path(output) if output else Path(f"calendar_{metric}_{year or 'current'}.html")
        
        html_content = visualizer.generate_calendar_html(
            calendar_data=sample_data,
            output_path=output_path,
            year=year
        )
        
        click.echo(f"Generated calendar visualization at {output_path}")
        
    except Exception as e:
        click.echo(f"Failed to generate calendar: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--output', type=click.Path(), help='Output HTML file path')
def time_series(output):
    """Generate time series visualization."""
    try:
        from .time_series_visualizer_stub import TimeSeriesVisualizer
        
        visualizer = TimeSeriesVisualizer()
        sample_data = visualizer.generate_sample_time_series_data()
        
        output_path = Path(output) if output else Path("time_series.html")
        
        html_content = visualizer.generate_diversity_trends_chart(
            time_series_data=sample_data,
            output_path=output_path
        )
        
        click.echo(f"Generated time series visualization at {output_path}")
        
    except Exception as e:
        click.echo(f"Failed to generate time series: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--output', type=click.Path(), help='Output HTML file path')
def dashboard(output):
    """Generate species dashboard."""
    try:
        from .dashboard_generator_stub import DashboardGenerator
        
        generator = DashboardGenerator()
        sample_data = generator.generate_sample_dashboard_data()
        
        output_path = Path(output) if output else Path("dashboard.html")
        
        html_content = generator.generate_species_distribution_chart(
            daily_summaries=sample_data["daily_summaries"],
            output_path=output_path
        )
        
        click.echo(f"Generated dashboard at {output_path}")
        
    except Exception as e:
        click.echo(f"Failed to generate dashboard: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--output', '-o', default='comparative_analysis.html', help='Output HTML file')
@click.option('--data-path', help='Path to daily summaries JSON file')
def generate_comparative_analysis(output: str, data_path: str):
    """Generate comparative analysis dashboard."""
    try:
        from ..analysis.comparative_analysis_stub import ComparativeAnalyzer
        from ..visualization.dashboard_generator import DashboardGenerator
        
        analyzer = ComparativeAnalyzer()
        dashboard_gen = DashboardGenerator()
        
        if data_path and Path(data_path).exists():
            import json
            with open(data_path, 'r') as f:
                daily_summaries = json.load(f)
        else:
            daily_summaries = analyzer._generate_sample_daily_summaries()
        
        results = analyzer.compare_temporal_diversity(daily_summaries)
        html = dashboard_gen.generate_comparative_analysis_dashboard(results, Path(output))
        
        click.echo(f"✅ Comparative analysis dashboard generated: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@cli.command()
@click.option('--output', '-o', default='functional_diversity.html', help='Output HTML file')
@click.option('--traits-path', help='Path to traits database JSON file')
def generate_functional_diversity(output: str, traits_path: str):
    """Generate functional diversity analysis dashboard."""
    try:
        from ..analysis.functional_diversity_stub import FunctionalDiversityAnalyzerStub as FunctionalDiversityAnalyzer
        from ..visualization.dashboard_generator import DashboardGenerator
        
        analyzer = FunctionalDiversityAnalyzer()
        dashboard_gen = DashboardGenerator()
        
        species_list = ["Taraxacum officinale", "Plantago major", "Trifolium repens", "Poa annua"]
        
        if traits_path and Path(traits_path).exists():
            import json
            with open(traits_path, 'r') as f:
                traits_data = json.load(f)
            analyzer.load_traits_from_dict(traits_data)
        else:
            sample_traits = analyzer.generate_sample_trait_database(species_list)
            analyzer.load_traits_from_dict(sample_traits)
        
        sample_abundances = {species: 10 - i*2 for i, species in enumerate(species_list)}
        results = analyzer.calculate_functional_diversity(sample_abundances)
        
        html = dashboard_gen.generate_functional_diversity_dashboard(results, Path(output))
        
        click.echo(f"✅ Functional diversity dashboard generated: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")

if __name__ == '__main__':
    cli()
