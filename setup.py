from setuptools import setup, find_packages

setup(
    name='mep',
    version='0.0.1',
    url='https://github.com/chc273/mep',
    packages=find_packages(),
    author='Chi Chen',
    author_email="chc273@eng.ucsd.edu",
    description='Find minimal energy path for atomistic systems',
    long_description='Find minimal energy path for atomistic systems',
    download_url='https://github.com/chc273/mep',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
    ],
)
