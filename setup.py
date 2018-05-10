from setuptools import setup

version = open('VERSION').read()

try:
    import pypandoc
    description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    description = open('README.md').read()

setup(
    packages = ['cluda'],

    install_requires = ['numpy'],
    zip_safe = False,

    name = 'cluda',
    version = version,
    description = '',
    long_description = description,

    url = 'https://github.com/tbenthompson/cluda',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any']
)
