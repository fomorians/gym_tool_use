from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

install_requires = ["gym", "pycolab", "gym-pycolab"]
with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="gym-tool-use",
    version="1.0.0",
    author="Fomoro AI",
    author_email="team@fomoro.com",
    description="Gym tool use environments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fomorians/gym_tool_use",
    download_url="https://github.com/fomorians/gym_tool_use/archive/v1.0.0.tar.gz",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    keywords=["gym", "pycolab", "reinforcement-learning", "tool-use"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
