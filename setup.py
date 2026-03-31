from setuptools import setup, find_packages

setup(
    name="pyAuger",
    version="1.0.0",
    description="Ab-initio direct Auger recombination calculator for semiconductors",
    author="Hamza Alhasan",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pymatgen",
    ],
)
