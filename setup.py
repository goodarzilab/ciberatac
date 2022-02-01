import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CiberATAC",
    version="0.0.1",
    author="Mehran Karimzadeh",
    author_email="mehran.karimzadeh@uhnresearch.ca",
    description="Deep learning model for deconvolving chromatin accessibility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goodarzilab/ciberatac",
    project_urls={
        "Bug Tracker": "https://github.com/goodarzilab/ciberatac/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "pytroch >= 1.4.0",
        "numpy >= 1.18.1",
        "apex",
        "scipy >= 1.5.4",
        "pandas >= 1.1.4",
        "sklearn >= 0.21.2",
        "pyBigWig >= 0.3.17",
        "seaborn >= 0.9.0"
    ]
)
