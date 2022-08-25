#!/bin/bash
set -e

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# core
python -m isort -rc iflearner/
python -m black --config pyproject.toml --line-length 88 iflearner/
python -m docformatter -i -r iflearner/

# examples
python -m isort -rc examples/
python -m black examples/
python -m docformatter -i -r examples/