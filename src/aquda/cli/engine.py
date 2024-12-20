import click
import sys
from typing import Any

from . import colour

@click.command()
@click.option('-gen', is_flag=True, default=False, help='Data generation mode')
@click.option('-aug', is_flag=True, default=False, help='Data augmentation mode')
@click.option('-validate', is_flag=True, default=False, help='Data validation mode')
@click.option('-l', '--lang', default=['en'], show_default=True, 
              multiple=True,
              help='Language to process')
def run_cli(gen: bool, aug: bool, validate: bool, lang: list[str]) -> int:
    print(f'Gen = {gen}')
    print(f'Aug = {aug}')
    print(f'Validate = {validate}')
    print(f'Lang = {lang}')
    run_mode = get_run_mode(gen, aug, validate)
    if run_mode < 0:
        return -1
    return 0

def get_run_mode(gen: bool, aug: bool, validate: bool) -> int:
    # One mode must be true
    if sum([gen, aug, validate]) != 1:
        sys.stderr.write(f'{colour.RED}ERROR: Run mode must be one of [-gen, -aug, -validate]{colour.DEFAULT}')
        return -1
    return 0