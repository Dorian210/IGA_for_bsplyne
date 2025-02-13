# Chemin du package
PACKAGE = ../IGA_for_bsplyne
# Dossier de sortie pour la documentation
DOC_DIR = .

.PHONY: all clean doc

# Commande par défaut : génère la doc
all: doc

# Génération de la documentation avec pdoc
doc:
	pdoc -o $(DOC_DIR) -d numpy $(PACKAGE)

# Nettoyage des fichiers générés (sans supprimer le Makefile)
clean:
	find $(DOC_DIR) -mindepth 1 ! -name 'Makefile' -exec rm -rf {} +