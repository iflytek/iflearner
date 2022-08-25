# introduce

## install dependencies
```shell
pip install -r requirements.txt
````

## Modify the content of the document
Built with the mkdocs tool, the document structure is defined in the mkdocs.yml file, and the specific content of the document is in the docs directory.

This document is edited in markdown syntax. If new documents need to be added, please edit mkdocs.yaml and add chapters.

## Local debugging documentation
```shell
mkdocs serve -a 127.0.0.1:8030
````
After executing the above command, you can view the content of the generated document through the http://127.0.0.1:8030 address.
> When the document is modified, the page content will be updated automatically.

## Local build documentation
```shell
mkdocs build
````
After executing the above command, the static files of the documentation site will be generated in the site directory, and the generated static files can be accessed by proxy.