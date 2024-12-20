import click
import sys
from typing import Any

from . import colour, run_modes

@click.command()
@click.option('-gen', is_flag=True, default=False, help='Data generation mode')
@click.option('-aug', is_flag=True, default=False, help='Data augmentation mode')
@click.option('-validate', is_flag=True, default=False, help='Data validation mode')
@click.option('--size', type=int, default=10, help='Size of dataset to generate or augment')
@click.option('-s', '--silence', is_flag=True, default=False, help='Only display final output.')
@click.option('-l', '--lang', default=['en'], show_default=True, 
              multiple=True,
              help='Language to process')
def run_cli(
    gen: bool, aug: bool, validate: bool, 
    lang: list[str], silence: bool,
    size: int) -> int:
    run_mode = get_run_mode(gen, aug, validate)
    if run_mode == run_modes.RunMode.UNKNOWN:
        return -1
    elif run_mode == run_modes.RunMode.GENERATOR:
        return run_generator(lang, silence, size)
    elif run_mode == run_modes.RunMode.AUGMENTOR:
        return run_augmentor(lang, silence, size)
    elif run_mode == run_modes.RunMode.VALIDATOR:
        return run_validator(lang, silence)
    return 0

def get_run_mode(gen: bool, aug: bool, validate: bool) -> run_modes.RunMode:
    # One mode must be true
    if sum([gen, aug, validate]) != 1:
        sys.stderr.write(f'{colour.RED}ERROR: Run mode must be one of [-gen, -aug, -validate]{colour.DEFAULT}')
        return run_modes.RunMode.UNKNOWN
    if gen:
        return run_modes.RunMode.GENERATOR
    if aug:
        return run_modes.RunMode.AUGMENTOR
    if validate:
        return run_modes.RunMode.VALIDATOR

def run_generator(lang: list[str], silence: bool, size: int):
    if not silence:
        print(f'Generating a dataset in {lang} of size {size}')
    pass

def run_validator(lang: list[str], silence: bool, size: int):
    pass

def run_augmentor(lang: list[str], silence: bool, size: int):
    pass