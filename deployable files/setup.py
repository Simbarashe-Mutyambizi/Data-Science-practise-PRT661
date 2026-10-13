# -*- coding: utf-8 -*-

from typing import List

from setuptools import find_packages, setup


def get_requirements(file_path: str) -> List[str]:
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = [
            req.strip()
            for req in file_obj.readlines()
            if req.strip() and req.strip() != "-e ."
        ]
    return requirements


setup(
    name="lthc_prevelance",
    version="0.0.1",
    author="Danala group 3 theme 2",
    author_email="simbarashewilliammutyambizi@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)