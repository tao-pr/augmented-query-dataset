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
@click.option('-merge', is_flag=True, default=False, help='Merge 2 or more JSON data files')
@click.option('--size', type=int, default=10, help='Size of dataset to generate or augment per language')
@click.option('--read', type=str, default=None, help='Specify an input query JSON file to process (UTF-8)')
@click.option('--write', type=str, default=None, help='Specify an output JSON file to write to (UTF-8)')
@click.option('--no-prompt-prefix', type=bool, default=False, is_flag=True,
              help='Do not add custom prefix text to my topic. Only used by -gen')
@click.option('-e', '--engine', type=click.Choice(['openai', 'spacy']), 
              default='spacy',
              help='Augmentation engine to use.' )
@click.option('-a', '--augmentor', multiple=True, type=click.Choice(query.PARAMS), 
              default=['lemma'], help='A linguistic technique to use for data augmentation.')
@click.option('-m', '--minput', type=str, default=[], multiple=True,
              help='Specify files to read and merge. Must be multiple.')
@click.option('-t', '--topic', default='sport', help='Context or topic to generate queries')
@click.option('-s', '--silence', is_flag=True, default=False, help='Only display final output.')
@click.option('-d', '--debug', is_flag=True, default=False, help='Enable interactive IPython prompt.')
@click.option('-l', '--lang', default=['en'], show_default=True, 
              multiple=True,
              help='Language to process')
def run_cli(
    gen: bool, aug: bool, validate: bool, merge: bool,
    lang: list[str], silence: bool,
    topic: str,
    read: str | None,
    write: str | None,
    minput: list[str],
    engine: str,
    augmentor: list[str],
    debug: bool,
    size: int,
    no_prompt_prefix: bool) -> int:
    run_mode = get_run_mode(gen, aug, validate, merge)

    lang = set(lang)
    augmentor = set(map(query.from_str, augmentor))

    if run_mode == run_modes.RunMode.UNKNOWN:
        return -1
    elif run_mode == run_modes.RunMode.GENERATOR:
        return run_generator(lang, silence, debug, size, topic, write, no_prompt_prefix)
    elif run_mode == run_modes.RunMode.AUGMENTOR:
        # Only translation with LLM takes language parameter
        if 'transl' not in augmentor or engine != 'openai':
            print(f'{colour.HIGHLIGHTED_GREY_LIGHT}WARNING:{colour.DEFAULT} Language parameter will be ignored. The languages from the input query dataset will be used.')
        return run_augmentor(silence, debug, size, read, write, engine, augmentor, lang)
    elif run_mode == run_modes.RunMode.VALIDATOR:
        return run_validator(lang, silence, debug)
    elif run_mode == run_modes.RunMode.MERGER:
        return run_merger(silence, debug, minput, write)
    return 0

def get_run_mode(gen: bool, aug: bool, validate: bool, merge: bool) -> run_modes.RunMode:
    # One and only mode must be true
    if sum([gen, aug, validate, merge]) != 1:
        sys.stderr.write(f'{colour.RED}ERROR: Run mode must be one of [-gen, -aug, -validate, -merge]{colour.DEFAULT}')
        return run_modes.RunMode.UNKNOWN
    if gen:
        return run_modes.RunMode.GENERATOR
    if aug:
        return run_modes.RunMode.AUGMENTOR
    if validate:
        return run_modes.RunMode.VALIDATOR
    if merge:
        return run_modes.RunMode.MERGER

def run_generator(lang: list[str], silence: bool, debug: bool, size: int, 
                  topic: str, output_path: str | None,
                  no_prompt_prefix: bool) -> query.QuerySet:
    client = agent.create()
    if not silence:
        print(f'Model to use: {os.environ.get(agent.OPENAI_MODEL)}')
        print(f'Generating {size} queries in {lang}')

    completion = agent.gen_queries(client, lang, size, topic, debug, silence, no_prompt_prefix)
    # taotodo handle failure
    out = completion.choices[0].message.parsed
    if debug:
        print(f'{colour.CYAN}Running run_generator, try inspecting out{colour.DEFAULT}')
        IPython.embed()
    
    # JSONify the response
    out_json = out.model_dump_json(indent = 2)
    print(out_json)
    if output_path is not None:
        output_path = os.path.expanduser(output_path)
        with open(output_path, 'w') as f:
            f.write(out_json)
    return out

def run_augmentor(silence: bool, debug: bool, size: int, 
                  input_path: str, output_path: str | None, 
                  engine: str, augmentor: set[query.VariantType],
                  lang: list[str] | None) -> query.QuerySet:
    input_path = os.path.expanduser(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'File not found: {input_path}')
    if not silence:
        print(f'Augmenting query file: {input_path} with {engine}')
    
    with open(input_path, 'r') as f:
        qs = query.QuerySet.model_validate(from_json(f.read()))

    # Collect all available languages of the input queries
    # and create augmentator of all those
    input_langs = set(map(lambda q: q.lang, qs.queries))
    if not silence:
        print(f'Languages to use: {input_langs}')

    aug = processor.get(engine, augmentor, input_langs, silence)

    if debug:
        print(f'{colour.CYAN}Running run_augmentor, try inspecting qs{colour.DEFAULT}')
        IPython.embed()

    # Process each query
    qvs = query.QueryVariantSet(queries = [])
    for q in qs.queries:
        completion = aug.process(q, size, debug, silence, lang)
        out = aug.parse_output(completion)
        if debug:
            print(f'{colour.CYAN}Augmenting query: {colour.DEFAULT}{q}')
            IPython.embed()
        qvs.queries += out.queries 

    out_json = qvs.model_dump_json(indent = 2)
    print(out_json)

    if output_path is not None:
        output_path = os.path.expanduser(output_path)
        with open(output_path, 'w') as f:
            f.write(out_json)

    return qvs

def run_merger(silence: bool, debug: bool, minput: list[str], write: str | None):
    if minput is None or len(minput) < 2:
        raise ValueError('Requiring 2 or more files to read from (via `-m` or `--merge` argument).')
    
    if write is None:
        raise ValueError('Requiring an output path via `--write` argument.')
    
    if not silence:
        print(f'{colour.CYAN}Merging files:{colour.DEFAULT} {', '.join(minput)}')
        print(f'{colour.CYAN}and write into:{colour.DEFAULT} {write}')

    merged = None
    for path in minput:
        with open(path, 'r') as f:
            q = query.QuerySet.model_validate(from_json(f.read()))
            if not silence:
                print(f'Merging {len(q.queries)} queries')
            if merged is None:
                merged = q
            else:
                merged.queries += q.queries
    
    path = os.path.expanduser(write)
    with open(path, 'w') as f:
        f.write(merged.model_dump_json(indent = 2))
    
    if not silence:
        print(f'Merged output written to {path}')

def run_validator(lang: list[str], silence: bool, debug: bool, size: int):
    pass