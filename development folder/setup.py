# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:16:49 2026

@author: SWRM
"""

#creating setup.py file
#responsible for creating the ml model and dashboard as a package

from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    #this function will return the list of requirements
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements        
setup(
      bane="lthc_prevelance",
      version='0.0.1',
      author='Danala group 3 theme 2',
      author_email='simbarashewilliammutyambizi@gmail.com',
      install_requires=get_requirements('requirements.txt')
  
      
      
      )