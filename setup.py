from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='metaX',
    version='1.0.1',
    license='GNU',
    author='Taufiqurrahman',
    author_email='metalearning@pps-ibrahimy.ac.id',
    url='https://github.com/metalearning-ibrahimy/IpyBibX',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'bertopic',
        'bert-extractive-summarizer',
        'chardet',
        'llmx',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'plotly',
        'scipy',
        'scikit-learn',
        'sentencepiece',
        'sentence-transformers',
        'squarify',
        'torch', 
        'torchvision',
        'torchaudio',
        'transformers',
        'umap-learn',
        'openai',
        'wordcloud'
    ],
    zip_safe=True,
    description='A Bibliometric and Scientometric Library Powered with Artificial Intelligence Tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
