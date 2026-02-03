import pdoc.doc
import os
import inspect
import sys
import argparse


def clean_text(text):
    """Nettoie et dédente les docstrings."""
    if not text:
        return ""
    return inspect.cleandoc(text)


def extract_module_content(mod, short=False):
    """Retourne le contenu formaté d'un module sans l'écrire sur disque."""
    lines = []
    lines.append(f"MODULE: {mod.fullname}")
    lines.append("=" * (len(mod.fullname) + 8))

    mod_doc = clean_text(mod.docstring)
    if not mod_doc or "Create a module object" in mod_doc:
        mod_doc = f"Documentation du module {mod.fullname}"
    lines.append(mod_doc + "\n")

    for cls in mod.classes:
        lines.append(f"CLASSE: {cls.name}")
        lines.append("-" * (len(cls.name) + 8))
        lines.append(clean_text(cls.docstring) + "\n")

        for m_obj in cls.own_members:
            if isinstance(m_obj, pdoc.doc.Function):
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
        lines.append("FONCTIONS GLOBALES:\n" + "~" * 18)
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
    """Génère les fichiers individuels (complets) et accumule le contenu global (court)."""
    # 1. On extrait la version courte pour le doc.txt global
    short_content = extract_module_content(mod, short=True)
    all_contents.append(short_content)

    # 2. On extrait la version complète pour le fichier individuel
    full_content = extract_module_content(mod, short=False)

    # Écriture du fichier individuel
    filename = os.path.join(output_folder, f"{mod.fullname}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"✓ Exporté (Complet) : {filename}")

    for submod in mod.submodules:
        process_recursive(submod, output_folder, all_contents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("-p", "--path", default="..")
    parser.add_argument("-o", "--output", default="api_txt")
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

    # Création du fichier doc.txt (sommaire complet)
    master_file = os.path.join(args.output, "doc.txt")
    with open(master_file, "w", encoding="utf-8") as f:
        f.write("================================================================\n")
        f.write(f"DOCUMENTATION COMPLÈTE DE L'API : {args.package.upper()}\n")
        f.write("================================================================\n\n")
        f.write("\n\n" + ("\n" + "#" * 64 + "\n").join(all_module_contents))

    print(f"\n✓ Succès : {master_file} généré.")


if __name__ == "__main__":
    main()
