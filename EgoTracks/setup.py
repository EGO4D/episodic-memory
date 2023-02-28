#!/usr/bin/env python3

from setuptools import find_packages, setup

PROJECTS = {
    "tracking.tools": "tools",
}

setup(
    name="egotracks",
    version="1.0",
    # author="",
    # url="unknown",
    # description="EgoTracks",
    # python_requires=">=3.7",
    # install_requires=[
    #     "matplotlib",
    #     "detectron2",
    #     "opencv-python",
    #     "pandas",
    #     "torchvision>=0.4.2",
    #     "scikit-learn",
    #     "iopath",
    # ],
    packages=find_packages(exclude=("tests", "tools")) + list(PROJECTS.keys()),
    package_dir=PROJECTS,
    package_data={"tracking.tools": ["**"]},
)
