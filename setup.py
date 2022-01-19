import setuptools
import subprocess


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="expdyn",
    version="0.0.1",
    author="Yushi Yang",
    author_email="yangyushi1992@icloud.com",
    description="Dynamical for experimental data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangyushi/expdyn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "expdyn"},  # location of Distribution package
    packages=setuptools.find_packages(where="expdyn"),  # find import package
    python_requires=">=3.5",
)
