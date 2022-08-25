from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import setuptools
import os

def _process_requirements():
    packages = open('requirements.txt').read().strip().split('\n')
    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            return_code = os.system('pip install {}'.format(pkg))
            assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
        else:
            requires.append(pkg)
    return requires

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iflearner",
    version="0.1.0",
    author="The iFLYTEK Turing Group",
    author_email="",
    description="Federated learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iflytek/iflearner",
    packages=[
        package for package in setuptools.find_packages()
        if package.startswith('iflearner')
    ],
    entry_points={
        "console_scripts": [
            "ifl_server = iflearner.business.homo.aggregate_server:main",
        ]
    },
    install_requires=_process_requirements(),
    setup_requires=[],
    license="Apache License 2.0",
    classifiers=[
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires='>=3.7',
)
