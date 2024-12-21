# Augmented Query Dataset

An experimental augmented query dataset generation with the NLP toolkit and LLM. The project packaging follows [PEP 621](https://peps.python.org/pep-0621/).

## Prerequisites

Generate OpenAI API key and store it in your environment variable `OPENAI_API_KEY`. The project when runs will look up from this variable. Make sure you have sufficient API credit.

To choose the OpenAI model for text generation, set your environment variable `OPENAI_API_MODEL`. See [OpenAI section](#openai) to find the model you prefer.

## Install and Run

Install locally from source (editable)

```sh
python -m install -e .
```

If one of the packages fail during installation, consider upgrading your build tools with

```sh
python -m pip install --upgrade pip setuptools wheel
```

Lookup the command arguments

```sh
python -m aquda --help
```

## Query Generation from LLM

Try generating your first search query

```sh
OPENAI_API_MODEL="gpt-4o-mini" python -m aquda -gen -lenglish -lgerman --size=1
```

or with debugging interactive IPython

```sh
OPENAI_API_MODEL="gpt-4o-mini" python -m aquda -gen -lenglish -lgerman --size=1 --debug
```

Generate queries for specific topic (turn off verbose)

```sh
OPENAI_API_MODEL="gpt-4o-mini" python -m aquda -gen -lenglish --size=5 \
--topic="christmas gift to buy in online store" -s
```

You can try crafting your prompt via topic parameter.

```sh
OPENAI_API_MODEL="gpt-4o-mini" python -m aquda -gen \
-lenglish -lgerman -lthai --size=15 \
--topic="outfits or clothings in online store. please also add brand specific to some search queries if possible, also try different types of queries from users from various demographical groups" --silence
```

## Query Augmentation

The augmentation picks the generated query set and apply the following linguistic methods.

- Lemmatization / Stemming
- Synonym Replacement
- POS Re-ordering
- Spelling Correction and Misspelling Variants
- Hypernyms and Hyponyms
- Acronym Variants
- Synonym by NER
- Language Translation Variants
- Adding / Replacing Modifiers

First of all, you need to generate the queries into a JSON file (don't forget to add `--silence` or `-s`). Then run aquda package in augmentation mode on this input.

```sh
# Generation to a JSON file
OPENAI_API_MODEL="gpt-4o-mini" python -m aquda -gen -lenglish --size=10 \
--topic="christmas gift to buy in online store" -s --write=query.json

# Run data augmentation with OpenAI API
OPENAI_API_MODEL="gpt-4o-mini" python -m aquda -aug \
--read=sample-queries/queries-en-de-christmas-gifts.json \
--engine=openai \
--write=augmented-queries-en-de-christmas-gifts.json
```

Augment the query dataset with hypernyms, hyponyms, synonym replacements with OpenAI

```sh
OPENAI_API_MODEL="gpt-4o-mini" python -m aquda -aug \
--read=sample-queries/queries-en-de-christmas-gifts.json \
--engine=openai \
-ahyponym -ahypernym -asyn-repl \
--size=10 \
--write=augmented-queries-en-de-christmas-gifts.json
```

## OpenAI

The project uses [OpenAI API](https://platform.openai.com/docs/overview) to generate an initial set of sample queries from a specific domain with various intents. As of the time of writing this, the [pricing](https://openai.com/api/pricing/) of the API is as listed below.

```
gpt-4o       $2.50 per 1 M input tokens
            $10.00 per 1 M output tokens (50% off for cached outputs)

gpt-4o mini $0.150 per 1 M input tokens
            $0.600 per 1 M output tokens (50% off for cached outputs)

o1          $15.00 per 1 M input tokens
            $60.00 per 1 M output tokens (50% off for cached outputs)
          
o1 mini     $3.00 per 1 M input tokens
           $12.00 per 1 M output tokens (50% off for cached outputs)
```

> NOTE: Batch API also adds 50% extra discount to both inputs and outputs.

Comparison of the models

```
gpt-4o        128K context    Knowledge of October 2023 cutoff
gpt-4o mini   n/a             Smarter than gpt-3.5 turbo
o1            200K context    Knowledge of October 2023 cutoff. Most powerful reasoning model.
                              supports structured inputs.
```

## Development Notes

[Spacy unfortunately fails to install on Python 3.13](https://github.com/explosion/spaCy/issues/13658) regardless of the Operating system due to their dependencies including `thic` and `srsly` don't build. This project therefore pins to python version 3.12.


## Licence

GNU
