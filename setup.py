#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

test_requirements = []

setup(
    author="Brian Clarke",
    author_email="brian.clarke@dkfz.de",
    python_requires=">=3.8",  # TODO: pick the right version
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    description="Integration of variant annotations using deep set networks boosts rare variant association genetics",
    entry_points={
        "console_scripts": [
            "deeprvat_config=deeprvat.deeprvat.config:cli",
            "deeprvat_train=deeprvat.deeprvat.train:cli",
            "deeprvat_associate=deeprvat.deeprvat.associate:cli",
            "deeprvat_preprocess=deeprvat.preprocessing.preprocess:cli",
            "deeprvat_evaluate=deeprvat.deeprvat.evaluate:evaluate",
            "seed_gene_pipeline=deeprvat.seed_gene_discovery.seed_gene_discovery:cli",
            "seed_gene_evaluate=deeprvat.seed_gene_discovery.evaluate:cli",
            "deeprvat_cv_utils=deeprvat.cv_utils:cli",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="deeprvat",
    name="deeprvat",
    packages=find_packages(include=["deeprvat", "deeprvat.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/PMBio/deeprvat",
    version="0.1.0",
    zip_safe=False,
)
