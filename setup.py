"""
lambdata - a collection of Data Science helper functions
"""
import setuptools
REQUIRED = [
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn"
]
with open("README.md", "r") as file:
    LONG_DESCRIPTION = file.read()
setuptools.setup(
    name="lambdata-skhabiri",
    version="0.0.1",
    author="skhabiri",
    description="fit estimate functions",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/skhabiri/lambdata",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=REQUIRED,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
