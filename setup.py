from setuptools import setup, find_packages

setup(
    name="sps",
    version="0.1",
    authors="Ibrahim Hroob, Benedikt Mersch",
    package_dir={"": "src"},
    description="Stable Points Segmentation",
    packages=find_packages(where="src"),
    install_requires=[
        "Click==7.0",
        "numpy==1.20.1",
        "pytorch_lightning==1.6.4",
        "PyYAML==6.0",
        "tqdm==4.62.3",
        "torch",
        "ninja",
    ],
)
