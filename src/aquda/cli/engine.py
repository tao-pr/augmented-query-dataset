import click
import IPython
import os
import sys
from pydantic_core import from_json
from typing import Any

from . import colour, run_modes
from ..openai import agent
from ..text import query, processor

@click.command()
@click.option('-gen', is_flag=True, default=False, help='Data generation mode')
@click.option('-aug', is_flag=True, default=False, help='Data augmentation mode')
@click.option('-validate', is_flag=True, default=False, help='Data validation mode')
@click.option('--size', type=int, default=10, help='Size of dataset to generate or augment per language')
@click.option('--file', type=str, help='Specify an input query JSON file to process')
@click.option('-e', '--engine', type=click.Choice(['openai', 'lib']), default='lib',
              help='Augmentation engine to use' )
@click.option('-a', '--augmentor', type=click.Choice(query.PARAMS), default='lemma',
              help='Pick one of the augmentation techniques')
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
    file: str,
    engine: str,
    augmentor: str,
    debug: bool,
    size: int) -> int:
    run_mode = get_run_mode(gen, aug, validate)
    if run_mode == run_modes.RunMode.UNKNOWN:
        return -1
    elif run_mode == run_modes.RunMode.GENERATOR:
        return run_generator(lang, silence, debug, size, topic)
    elif run_mode == run_modes.RunMode.AUGMENTOR:
        return run_augmentor(lang, silence, debug, size, file, engine, augmentor)
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

    completion = agent.gen_queries(client, lang, size, topic, debug, silence)
    # taotodo handle failure
    out = completion.choices[0].message.parsed
    if debug:
        print(f'{colour.CYAN}Running run_generator, try inspecting out{colour.DEFAULT}')
        IPython.embed()
    
    # JSONify the response
    print(out.model_dump_json(indent = 2))
    return out

def run_augmentor(lang: list[str], silence: bool, debug: bool, size: int, 
                  file: str, engine: str, augmentor: str) -> query.QuerySet:
    fullpath = os.path.expanduser(file)
    aug = processor.get(engine, augmentor, lang)

    if not os.path.exists(fullpath):
        raise FileNotFoundError(f'File not found: {fullpath}')
    if not silence:
        print(f'Augmenting query file: {fullpath} with {engine} ({aug})')
    
    with open(fullpath, 'r') as f:
        qs = query.QuerySet.model_validate(from_json(f.read()))

    if debug:
        print(f'{colour.CYAN}Running run_augmentor, try inspecting qs{colour.DEFAULT}')
        IPython.embed()

    # Process each query
    qvs = query.QueryVariantSet(queries = [])
    for q in qs.queries:
        completion = aug.process(q, size, debug, silence)
        out = completion.choices[0].message.parsed
        if debug:
            print(f'{colour.CYAN}Augmenting query: {colour.DEFAULT}{q}')
            IPython.embed()
        qvs.queries += out.queries 

    print(qvs.model_dump_json(indent = 2))
    return qvs

def run_validator(lang: list[str], silence: bool, debug: bool, size: int):
    pass