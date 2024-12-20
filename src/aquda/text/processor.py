import os
from collections.abc import Callable

from ..openai import agent
from ..text import query

class Augmentor(object):
    def process(self, query: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> str:
        pass

class OpenAIAugmentor(Augmentor):
    def __init__(self, client: object, vtype: query.VariantType, lang: list[str]):
        self.client = client
        self.vtype = vtype
        self.lang = lang

    def process(self, query: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> str:
        return agent.augment_query(
            self.client,
            query,
            self.vtype,
            num,
            debug,
            silence
        )

class LibAugmentor(Augmentor):
    def __init__(self, vtype: query.VariantType, lang: list[str]):
        self.vtype = vtype
        self.lang = lang

    def process(self, query: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> str:
        # taotodo
        return None

# lang is only used for language translation mode
def get(engine: str, augmentor: str, lang: list[str]) -> Augmentor:
    if engine == 'lib':
        return LibAugmentor(augmentor, lang)
    elif engine == 'openai':
        client = agent.create()
        return OpenAIAugmentor(client, augmentor, lang)
    else:
        raise ValueError(f'Unknown augmentation engine: {engine}')
