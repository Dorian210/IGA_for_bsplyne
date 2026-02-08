import pdoc.doc
import os
import inspect
import sys
import argparse


def clean_text(text):
    """Cleans and dedents docstrings."""
    if not text:
        return ""
    return inspect.cleandoc(text)


def extract_module_content(mod, short=False):
    """Returns the formatted content of a module without writing it to disk."""
    lines = []
    lines.append(f"MODULE: {mod.fullname}")
    lines.append("=" * (len(mod.fullname) + 8))

    mod_doc = clean_text(mod.docstring)
    # Default text if docstring is empty or generic
    if not mod_doc or "Create a module object" in mod_doc:
        mod_doc = f"Documentation for module {mod.fullname}"
    lines.append(mod_doc + "\n")

    for cls in mod.classes:
        lines.append(f"CLASS: {cls.name}")
        lines.append("-" * (len(cls.name) + 8))
        lines.append(clean_text(cls.docstring) + "\n")

        for m_obj in cls.own_members:
            if isinstance(m_obj, pdoc.doc.Function):
                # Include public methods and specific dunder methods
                if not m_obj.name.startswith("_") or m_obj.name in [
                    "__init__",
                    "__call__",
                ]:
                    lines.append(f"  >>> {m_obj.name}{str(m_obj.signature)}")
                    if m_obj.docstring and not short:
                        doc_body = clean_text(m_obj.docstring)
                        indented_doc = "\n".join(
                            "      " + line for line in doc_body.splitlines()
                        )
                        lines.append(indented_doc)
                    lines.append("")
        lines.append("")

    functions = [m for m in mod.own_members if isinstance(m, pdoc.doc.Function)]
    if functions:
        lines.append("GLOBAL FUNCTIONS:\n" + "~" * 18)
        for func in functions:
            if not func.name.startswith("_"):
                lines.append(f"  >>> {func.name}{str(func.signature)}")
                if func.docstring and not short:
                    doc_body = clean_text(func.docstring)
                    indented_doc = "\n".join(
                        "      " + line for line in doc_body.splitlines()
                    )
                    lines.append(indented_doc)
                lines.append("")

    return "\n".join(lines)


def process_recursive(mod, output_folder, all_contents):
    """Generates individual full files and accumulates global short content."""
    # 1. Extract short version for the global doc.txt
    short_content = extract_module_content(mod, short=True)
    all_contents.append(short_content)

    # 2. Extract full version for the individual file
    full_content = extract_module_content(mod, short=False)

    # Write individual file
    filename = os.path.join(output_folder, f"{mod.fullname}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"✓ Exported (Full): {filename}")

    for submod in mod.submodules:
        process_recursive(submod, output_folder, all_contents)


def main():
    parser = argparse.ArgumentParser(
        description="Generate plain text API documentation."
    )
    parser.add_argument("package", help="The name of the package to document")
    parser.add_argument(
        "-p", "--path", default="..", help="Path to the package directory"
    )
    parser.add_argument("-o", "--output", default="api_txt", help="Output folder name")
    args = parser.parse_args()

    package_dir = os.path.abspath(args.path)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Extraction
    root_module = pdoc.doc.Module.from_name(args.package)

    all_module_contents = []
    process_recursive(root_module, args.output, all_module_contents)

    # Create the master doc.txt (full summary)
    master_file = os.path.join(args.output, "doc.txt")
    with open(master_file, "w", encoding="utf-8") as f:
        f.write("================================================================\n")
        f.write(f"FULL API DOCUMENTATION: {args.package.upper()}\n")
        f.write("================================================================\n\n")
        f.write("\n\n" + ("\n" + "#" * 64 + "\n").join(all_module_contents))

    print(f"\n✓ Success: {master_file} generated.")


if __name__ == "__main__":
    main()
