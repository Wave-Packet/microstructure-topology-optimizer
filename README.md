# Optimisation topologique de microstructures periodiques

Ce depot rassemble le code Python/DOLFINx developpe pendant le stage consacré
a la conception optimale de microstructures. Le tenseur d'elasticite
homogeneise est calcule par elements finis, puis la geometrie est mise a jour
par une methode level set et une derivee topologique.

## Installation avec Conda/Mamba

L'environnement est entierement decrit dans `environment.yml`. Mamba est
recommande car la resolution des dependances DOLFINx est plus rapide :

```bash
mamba env create -f environment.yml
mamba activate microstructure-topology
python scripts/check_environment.py
pytest -q
```

Avec Conda, seule la premiere commande change :

```bash
conda env create -f environment.yml
```

## Choisir la fonction objectif

Afficher les criteres disponibles sans initialiser DOLFINx :

```bash
python Optimus.py --list-criteria
```

Lancer par exemple la maximisation du module de compression :

```bash
python Optimus.py --criterion bulk_modulus_max --no-plot
```

Les sept fonctions de l'article sont disponibles :

| Option `--criterion` | Grandeur optimisee | Penalite volumique par defaut |
|---|---|---:|
| `horizontal_rigidity_max` | rigidite horizontale maximale | 10 |
| `bulk_modulus_max` | module de compression maximal | 20 |
| `shear_modulus_max` | module de cisaillement maximal | 50 |
| `modified_poisson_ratio_min` | coefficient de Poisson modifie minimal | 0 |
| `modified_poisson_ratio_max` | coefficient de Poisson modifie maximal | 0 |
| `poisson_ratio_min` | coefficient de Poisson minimal | 0 |
| `poisson_ratio_max` | coefficient de Poisson maximal | 0 |

La valeur peut etre remplacee avec `--volume-penalty`. Le comportement initial
du programme correspond a `modified_poisson_ratio_max`.

Pour un essai rapide sans fenetre graphique :

```bash
python Optimus.py \
  --criterion modified_poisson_ratio_max \
  --mesh-resolution 8 \
  --max-iterations 1 \
  --no-plot
```

La resolution normale du maillage reste 62. Les fichiers `mesh.xdmf`,
`mesh.h5`, `solution.xdmf` et `solution.h5` sont des sorties de calcul et ne
sont pas versionnes.

## Afficher les resultats

Commencer par verifier l'environnement :

```bash
mamba env create -f environment.yml
mamba activate microstructure-topology
python scripts/check_environment.py
```

### Affichage direct avec PyVista

Pour tester rapidement le programme avec une petite resolution et afficher la
microstructure apres chaque iteration :

```bash
python Optimus.py \
  --criterion modified_poisson_ratio_max \
  --mesh-resolution 8 \
  --max-iterations 5
```

Une fenetre PyVista s'ouvre apres chaque iteration. Fermer la fenetre permet au
calcul de poursuivre l'iteration suivante. Il est conseille de commencer avec
une resolution de 8, puis d'essayer 20 avant d'utiliser la resolution normale
de 62.

Un calcul complet peut par exemple etre lance avec :

```bash
python Optimus.py \
  --criterion modified_poisson_ratio_max \
  --mesh-resolution 62 \
  --max-iterations 100
```

### Visualisation avec ParaView

Pour executer le calcul sans fenetre graphique et conserver sa sortie dans un
journal :

```bash
python Optimus.py \
  --criterion bulk_modulus_max \
  --mesh-resolution 20 \
  --max-iterations 30 \
  --no-plot | tee optimisation.log
```

Le programme cree les fichiers suivants dans le repertoire courant :

- `mesh.xdmf` et `mesh.h5` : maillage ;
- `solution.xdmf` et `solution.h5` : champ final du module de Young ;
- `optimisation.log`, lorsque `tee` est utilise : matrices homogenisees,
  objectif `J`, aire, angle `theta` et coefficient de Poisson.

Dans ParaView, ouvrir `solution.xdmf`, cliquer sur **Apply**, choisir la
variable `Y` dans **Coloring**, puis selectionner **Surface With Edges** pour
voir la microstructure. Les fichiers de sortie sont remplaces a chaque
iteration : ils correspondent donc au dernier etat calcule.

### Resultats des controles GitHub

Le workflow GitHub Actions verifie l'installation, les formules des sept
criteres et un petit calcul DOLFINx. Son etat et ses journaux sont disponibles
dans l'onglet **Actions** du depot. Ce workflow valide le code, mais ne publie
pas actuellement d'image de la microstructure.

## Verifier les formules sans DOLFINx

`criteria.py` ne depend que de NumPy. Les tests comparent les derivees
analytiques des sept criteres a des differences finies :

```bash
pytest -q tests/test_criteria.py
```

## Conteneur de secours

Un `Dockerfile` fonde sur l'image DOLFINx-MPC 0.10.1 est fourni si une machine
ne peut pas installer l'environnement Conda/Mamba.

## Integration continue

Le workflow `.github/workflows/conda.yml` recree l'environnement avec
Micromamba a chaque push ou pull request. Il controle les imports scientifiques,
les sept derivees, l'interface de commande et un calcul DOLFINx de petite taille.
