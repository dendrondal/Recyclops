from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    packages =['cif3r', 'app']
    version="0.1.0",
    description="Binary image classificether an image is of a recycleable or compostable object",
    author="Dal Williams",
    license="MIT",
)
