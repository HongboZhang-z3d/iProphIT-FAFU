#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='iprophit',
    version='1.0.0',
    packages=find_packages(),
    python_requires='>=3.12,<3.13',
    
    entry_points={
        'console_scripts': [
            'iprophit=iprophit.classifier:main',
        ],
    },
    
    install_requires=[
        'torch>=2.6.0',
        'biopython>=1.86',
        'numpy>=2.0,<2.4',
        'pandas>=1.5.0',
        'tqdm>=4.60.0',
        'requests',  # 用于下载权重
    ],
    
    author='Hongbo Zhang',
    description='Deep learning approach for identifying inducible prophage activity',
    url='https://github.com/HongboZhang-z3d/iProphIT-FAFU',
    license='GPL-3.0',
)
