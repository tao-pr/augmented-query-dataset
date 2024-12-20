# Augmented Query Dataset

An experimental augmented query dataset generation with the NLP toolkit and LLM. The project packaging follows [PEP 621](https://peps.python.org/pep-0621/), however it is not yet 

## Prerequisites

TBD

## Install and Run

Install locally from source (editable)

```sh
python -m install -e .
```

Lookup the command arguments

```sh
python -m aquda --help
```

## Query Generation from LLM

TBD

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

### API Key

Generate OpenAI API key and store it in your environment variable `OPENAI_API_KEY`. The project when runs will look up from this variable. Make sure you have sufficient API credit.

## Licence

GNU
