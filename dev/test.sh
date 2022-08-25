#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo $PWD
echo "=== test.sh ==="

python -m isort iflearner/ examples/  && echo "- isort:         done" &&
python -m docformatter -rc iflearner/  examples/   && echo "- docformatter:  done" &&
python -m black  --check --line-length 88 iflearner/  examples/      && echo "- black:         done" &&
python -m mypy  --config-file mypy.ini iflearner/     && echo "- mypy:          done" &&
python -m flake8 iflearner/ && echo "- flake8:        done" &&
echo "- All Python checks passed"