import os
from collections.abc import Callable

from ..openai import agent
from ..spacy import wordnet
from ..text import query
from ..cli import colour

class Augmentor(object):
    def process(self, query: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> str:
        pass

    def parse_output(self, output: object) -> query.QueryVariantSet:
        pass

# Generative LLM
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
    
    def parse_output(self, completion: object) -> query.QueryVariantSet:
        # taotodo handle error
        return completion.choices[0].message.parsed

# Transformer-based (mainly RoBERTa)
class SpacyAugmentor(Augmentor):
    # NOTE: supports only 1 language, unlike other augmentors
    def __init__(self, api: object, vtypes: set[query.VariantType], lang: str):
        self.api = api
        self.vtypes = vtypes
        self.lang = lang
        self.VMAP: dict[str, Callable[[str, object, str], query.QueryVariant]] = {
            'lemma': wordnet.lemmatize
        }
        self.SUPPORTED_VTYPES = self.VMAP.keys()

    def process(self, q: query.Query, num: int, debug: bool, silence: bool, **kwargs) -> query.QueryVariant:
        if any([v not in self.SUPPORTED_VTYPES for v in self.vtypes]):
            raise ValueError(f'Some variant types are not supported by Spacy. It only supports any of {[v.name for v in self.SUPPORTED_VTYPES]}. (Got {self.vtypes} instead)')
        
        output_variant = None
        if not silence:
            print('────────────────')
        for vtype in self.vtypes:
            # taotodo should instead take [lang] from original query
            if not silence:
                print(f'{colour.CYAN}Applying {vtype}, lang={self.lang}: {colour.DEFAULT}{q.original}')
            processed = self.VMAP[vtype](self.lang, self.api, q.original)
            
            if output_variant is None:
                output_variant = processed
            else:
                output_variant.variants += processed.variants

        if not silence:
            print('────────────────')
        return output_variant
    
    def parse_output(self, output: query.QueryVariant) -> query.QueryVariantSet:
        return query.QueryVariantSet(
            queries = [output]
        )

# lang is only used for language translation mode
def get(engine: str, vtypes: set[query.VariantType], lang: set[str], silence: bool) -> Augmentor:
    if engine.startswith('spacy-'):
        model = engine.split('-')[-1] # suffix of the engine will be used as a suffix of spacy model
        
        # Only supports a single language
        if len(lang) != 1:
            raise ValueError(f'Spacy models support only 1 language at a time. (Got {lang} instead)')

        lang = ''.join(lang)
        api = wordnet.create(model, lang, silence)
        return SpacyAugmentor(api, vtypes, lang)
    elif engine == 'openai':
        client = agent.create()
        return OpenAIAugmentor(client, vtypes, lang)
    else:
        raise ValueError(f'Unknown augmentation engine: {engine}')
