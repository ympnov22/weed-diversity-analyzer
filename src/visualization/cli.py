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
        from .calendar_visualizer import CalendarVisualizer
        
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
        from .time_series_visualizer import TimeSeriesVisualizer
        
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
        from .dashboard_generator import DashboardGenerator
        
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


if __name__ == '__main__':
    cli()
