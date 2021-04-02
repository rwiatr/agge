from setuptools import setup, find_packages

setup(
    name='agge',
    version='0.0.1',
    description='Aggregate Encoders',
    long_description_content_type='text/markdown',
    url='https://github.com/rwiatr/agge',
    author='Roman Wiatr',
    keywords='aggregate encoders, logistic regression, CTR',

    package_dir={'': 'src'},
    packages=find_packages(where='src'),  # Required
    python_requires='=3.8',

    install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib'],

    project_urls={
        'Paper': 'TBA',
        'Source': 'https://github.com/rwiatr/agge/',
    },
)