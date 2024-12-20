import click
import IPython
import json
import os
import sys
from typing import Any

from . import colour, run_modes
from ..openai import agent
from ..text import query

@click.command()
@click.option('-gen', is_flag=True, default=False, help='Data generation mode')
@click.option('-aug', is_flag=True, default=False, help='Data augmentation mode')
@click.option('-validate', is_flag=True, default=False, help='Data validation mode')
@click.option('--size', type=int, default=10, help='Size of dataset to generate or augment per language')
@click.option('-t', '--topic', default='sport', help='Context or topic to generate queries')
@click.option('-s', '--silence', is_flag=True, default=False, help='Only display final output.')
@click.option('-d', '--debug', is_flag=True, default=False, help='Enable interactive IPython prompt.')
@click.option('-l', '--lang', default=['en'], show_default=True, 
              multiple=True,
              help='Language to process')
def run_cli(
    gen: bool, aug: bool, validate: bool, 
    lang: list[str], silence: bool,
    topic: str,
    debug: bool,
    size: int) -> int:
    run_mode = get_run_mode(gen, aug, validate)
    if run_mode == run_modes.RunMode.UNKNOWN:
        return -1
    elif run_mode == run_modes.RunMode.GENERATOR:
        return run_generator(lang, silence, debug, size, topic)
    elif run_mode == run_modes.RunMode.AUGMENTOR:
        return run_augmentor(lang, silence, debug, size)
    elif run_mode == run_modes.RunMode.VALIDATOR:
        return run_validator(lang, silence, debug)
    return 0

def get_run_mode(gen: bool, aug: bool, validate: bool) -> run_modes.RunMode:
    # One and only mode must be true
    if sum([gen, aug, validate]) != 1:
        sys.stderr.write(f'{colour.RED}ERROR: Run mode must be one of [-gen, -aug, -validate]{colour.DEFAULT}')
        return run_modes.RunMode.UNKNOWN
    if gen:
        return run_modes.RunMode.GENERATOR
    if aug:
        return run_modes.RunMode.AUGMENTOR
    if validate:
        return run_modes.RunMode.VALIDATOR

def run_generator(lang: list[str], silence: bool, debug: bool, size: int, topic: str) -> query.QuerySet:
    client = agent.create()
    if not silence:
        print(f'Model to use: {os.environ.get(agent.OPENAI_MODEL)}')
        print(f'Generating {size} queries in {lang}')

    completion = agent.gen_queries(client, lang, size, topic, debug)
    # taotodo handle failure
    out = completion.choices[0].message.parsed
    if debug:
        IPython.embed()
    
    # JSONify the response
    print(out.model_dump())
    return out

def run_validator(lang: list[str], silence: bool, debug: bool, size: int):
    pass

def run_augmentor(lang: list[str], silence: bool, debug: bool, size: int):
    pass