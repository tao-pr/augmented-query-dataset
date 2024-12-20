import os
from openai import OpenAI
from typing import Any

from ..text import query
from ..cli import colour

OPENAI_KEY = "OPENAI_API_KEY"
OPENAI_MODEL = "OPENAI_API_MODEL"

def create() -> object:
    return OpenAI(
        # It's default auth method, but wanna make it obvious.
        api_key = os.environ.get(OPENAI_KEY),
    )

def make_prompts(lang: str, num: int, topic: str, debug: bool, silence: bool) -> list[dict[str, str]]:
    # just simple prompt
    content = f'Generate {num} sample search queries in {', '.join(lang)} language the real user would use to search for {topic}'
    if not silence:
        print(f'{colour.CYAN}{colour.BOLD}Prompt:{colour.DEFAULT} {content}')
    return [{
        'role': 'developer',
        'content': content
    }]

def make_augmenting_prompts(num: int, orig: str, method: str, debug: bool, silence: bool) -> list[dict[str, str]]:
    # just simple prompt
    content = f'Generate {num} additional search queries from "{orig}" by applying {method}'
    if not silence:
        print(f'{colour.CYAN}{colour.BOLD}Prompt:{colour.DEFAULT} {content}')
    return [{
        'role': 'developer',
        'content': content
    }]

def gen_queries(client: object, lang: str, num: int, topic: str, debug: bool, silence: bool) -> list[query.Query]:
    # https://platform.openai.com/docs/api-reference/introduction
    model = os.environ.get(OPENAI_MODEL)
    if model is None:
        # No we don't decide the fallback model for anyone
        raise ValueError(f'Missing model name in env var: {OPENAI_MODEL}')
    completion = client.beta.chat.completions.parse(
        model = model,
        messages = make_prompts(lang, num, topic, debug, silence),
        response_format = query.QuerySet
    )
    return completion

def augment_query(client: object, orig: query.Query, typ: query.VariantType, 
                  num: int, debug: bool, silence: bool) -> list[query.QueryVariant]:
    model = os.environ.get(OPENAI_MODEL)
    if model is None:
        # No we don't decide the fallback model for anyone
        raise ValueError(f'Missing model name in env var: {OPENAI_MODEL}')
    completion = client.beta.chat.completions.parse(
        model = model,
        messages = make_augmenting_prompts(num, orig.original, typ, debug, silence),
        response_format = query.QueryVariantSet
    )
    return completion

