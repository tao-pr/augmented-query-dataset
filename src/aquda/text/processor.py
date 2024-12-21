import os
from collections.abc import Callable

from ..openai import agent
from ..text import query

class Augmentor(object):
    def process(self, query: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> str:
        pass

class OpenAIAugmentor(Augmentor):
    def __init__(self, client: object, vtypes: set[query.VariantType], lang: set[str]):
        self.client = client
        self.vtypes = vtypes
        self.lang = lang

    def process(self, query: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> str:
        return agent.augment_query(
            self.client,
            query,
            self.vtypes,
            num,
            debug,
            silence
        )

class LibAugmentor(Augmentor):
    def __init__(self, vtypes: set[query.VariantType], lang: set[str]):
        self.vtypes = vtypes
        self.lang = lang

    def process(self, query: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> str:
        # taotodo
        return None

# lang is only used for language translation mode
def get(engine: str, vtypes: set[query.VariantType], lang: set[str]) -> Augmentor:
    if engine == 'lib':
        return LibAugmentor(vtypes, lang)
    elif engine == 'openai':
        client = agent.create()
        return OpenAIAugmentor(client, vtypes, lang)
    else:
        raise ValueError(f'Unknown augmentation engine: {engine}')
