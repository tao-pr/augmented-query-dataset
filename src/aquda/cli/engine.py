import click

@click.command()
@click.option('--lang', default='en', help='Language to process')
def run_cli(lang):
    print(f'Lang = {lang}')
    return None