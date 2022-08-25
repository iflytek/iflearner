#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

rm -rf dist iflearner.egg-info || true
python setup.py sdist

# whl
## python setup.py bdist_wheel