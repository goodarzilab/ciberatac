import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CiberATAC",
    version="0.0.1",
    author="Mehran Karimzadeh",
    author_email="mehran.karimzadeh@uhnresearch.ca",
    description="Deep learning model for deconvolving chromatin accessibility data",
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
)
