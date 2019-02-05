from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages


setup(
    name='gym_tool_use',
    version='0.0.0',
    description='Gym tool-use environments.',
    url='https://github.com/fomorians/gym_tool_use',
    packages=find_packages(),
    install_requires=['pycolab', 'gym', 'gym_pycolab'],
    dependency_links=[
        'https://github.com/fomorians/gym_pycolab/tarball/master#egg=gym_pycolab-0.0.0'])