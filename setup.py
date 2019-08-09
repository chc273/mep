from setuptools import setup, find_packages
import os

DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, "README.md"), "r") as f:
    long_description = f.read()

setup(
    name='mep',
    version='0.0.1',
    url='https://github.com/chc273/mep',
    packages=find_packages(),
    author='Chi Chen',
    author_email="chc273@eng.ucsd.edu",
    description='Minimal energy path tools for atomistic systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url='https://github.com/chc273/mep',
    keywords=["materials", "science", "nudged elastic band"],
    license='BSD',
    install_requires=["numpy >= 1.9.0", "pymatgen", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],


)
