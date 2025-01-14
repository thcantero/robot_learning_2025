# setup.py
from setuptools import setup

setup(
    name='roble',
    version='0.2.0',
    packages=['roble'],
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read()
)