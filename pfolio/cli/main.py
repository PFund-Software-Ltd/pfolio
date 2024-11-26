import click
from trogon import tui

# from pfolio.cli.commands.PLACEHOLDER import ...


@tui(command='tui', help="Open terminal UI")
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
@click.version_option()
def pfolio_group(ctx):
    """PFolio's CLI"""
    ctx.ensure_object(dict)
    # ctx.obj['config'] = 


# pfolio_group.add_command(...)
