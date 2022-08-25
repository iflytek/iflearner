#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../


#First, Configture setuptools tools
# Linux
## vim ~/.pypirc
# Windows
## C:\Users\Username\.pypirc
: <<'COMMENT'
 [distutils]
 index-servers=pypi

 [pypi]
 repository=https://upload.pypi.org/legacy/
 username=
 password=
COMMENT

# Second, Upload to pypi
twine upload dist/* -r pypi
