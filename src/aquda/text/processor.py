from collections.abc import Callable
from pydantic_core import from_json

from ..openai import agent
from ..spacy import wordnet, conf
from ..text import query
from ..cli import colour

class Augmentor(object):
    def process(self, query: query.Query, num: int, 
                debug: bool, silence: bool, lang: set[str] | None, **kwargs) -> str:
        pass

    def parse_output(self, output: object) -> query.QueryVariantSet:
        pass

# Generative LLM
class OpenAIAugmentor(Augmentor):
    def __init__(self, client: object, vtypes: set[query.VariantType], lang: set[str]):
        self.client = client
        self.vtypes = vtypes
        self.lang = lang

    def process(self, query: query.Query, num: int, 
                debug: bool, silence: bool, lang: set[str] | None, **kwargs) -> str:
        return agent.augment_query(
            self.client,
            query,
            self.vtypes,
            num,
            debug,
            silence,
            lang
        )
    
    def parse_output(self, completion: object) -> query.QueryVariantSet:
        # taotodo handle error
        return completion.choices[0].message.parsed

# Transformer-based (mainly RoBERTa)
class SpacyAugmentor(Augmentor):
    # NOTE: supports only 1 language, unlike other augmentors
    def __init__(self, apis: dict[str, object], vtypes: set[query.VariantType]):
        self.apis = apis
        self.vtypes = vtypes
        self.VMAP: dict[str, Callable[[str, object, str], query.QueryVariant]] = {
            'lemma': wordnet.lemmatize,
            'syn-repl': wordnet.synonym_repl,
        }
        self.SUPPORTED_VTYPES = self.VMAP.keys()

    def process(self, q: query.Query, num: int, 
                debug: bool, silence: bool,
                lang: set[str] | None, **kwargs) -> query.QueryVariant:
        if any([v not in self.SUPPORTED_VTYPES for v in self.vtypes]):
            raise ValueError(f'Some variant types are not supported by Spacy. It only supports any of {self.SUPPORTED_VTYPES}. (Got {self.vtypes} instead)')
        
        output_variant = None
        if not silence:
            print('────────────────')
        for vtype in self.vtypes:
            if not silence:
                print(f'{colour.CYAN}Applying {vtype}, lang={q.lang}: {colour.DEFAULT}{q.original}')
            processed = self.VMAP[vtype](q.lang, self.apis[q.lang], q.original)

            if debug:
                print(f'Processing variant = {vtype}, try inspecting `q`, `processed`')
                import IPython
                IPython.embed()
            
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
def get(engine: str, vtypes: set[query.VariantType], 
        lang: set[str], silence: bool) -> Augmentor:
    if engine == 'spacy':
        with open('spacy.conf', 'r') as f:
            config = conf.Conf.model_validate(from_json(f.read()))
        
        # load Spacy models, one per language
        api_by_lang: dict[str, object] = {lng: wordnet.create(config.langs[lng].model, \
                                                              silence) \
                                                                for lng in lang}
        
        return SpacyAugmentor(api_by_lang, vtypes)
    elif engine == 'openai':
        client = agent.create()
        return OpenAIAugmentor(client, vtypes, lang)
    else:
        raise ValueError(f'Unknown augmentation engine: {engine}')
