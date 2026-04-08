from setuptools import setup, find_packages

setup(
    name="spectral-graph-patternboost",
    version="0.1.0",
    description="Geometric spectral graph design via PatternBoost",
    author="Christophe Pennetier",
    author_email="cpennetier@gmail.com",
    url="https://github.com/cpennetier/spectral-graph-patternboost",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.0",
    ],
)
