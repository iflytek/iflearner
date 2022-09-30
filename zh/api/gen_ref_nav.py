"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("../iflearner").glob("**/*.py")):
    path = Path(path)
    module_path = path.relative_to("../").with_suffix("")
    doc_path = path.relative_to("../", "iflearner").with_suffix(".md")
    full_doc_path = Path("docs/api/reference", doc_path)
    parts = list(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    elif parts[-1].endswith("_test"):
        continue
    nav[parts] = doc_path

    with mkdocs_gen_files.open(full_doc_path.absolute(), "w+") as fd:
        ident = ".".join(parts)
        fd.write("::: " + ident)
    mkdocs_gen_files.set_edit_path(full_doc_path,  path)

# with mkdocs_gen_files.open(Path("docs/api/reference/SUMMARY.md").absolute(), "w+") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
