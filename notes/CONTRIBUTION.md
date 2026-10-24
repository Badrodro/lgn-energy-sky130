# Contribution scientifique la plus défendable — analyse (23/10/2026)

Règle : une observation déjà publiée n'est pas présentée comme nouvelle. Chaque ligne indique
l'état de l'art qui la précède.

## 1. Ce qui est DÉJÀ CONNU (à citer, pas à revendiquer)

| observation | déjà montrée par |
|---|---|
| la tête de sortie (popcount/GroupSum) coûte autant ou plus que le réseau | LILogicNet (~75 % des LUT FPGA), BitLogic (59 k contre 61 k NAND2-GE) |
| l'encodage thermomètre est un coût caché | DWN (jusqu'à 3,2x de LUT avec encodeur sur puce) |
| câblage aléatoire difficile à router ; le câblage structuré réduit la longueur | Fieldhouse 2026 (non routable en SKY130), FPGN (3,5x) |
| apprendre les connexions réduit fortement le nombre de portes | Mommen et al. 2025/2026, LILogicNet |
| le nombre de seuils a un effet limité | LILogicNet (CIFAR-10, « threshold count has limited effect ») |
| DLGN implantés en SKY130 jusqu'au GDSII et fabriqués sur navettes | Fieldhouse 2026 ; Zioma & Armaitis (Tiny Tapeout, IHP, GF180) |

## 2. Ce que nous pouvons revendiquer (avec le niveau de preuve)

### C-A — Comptabilité de coût de bout en bout, ouverte et reproductible, au niveau ASIC (principale)

Chaîne entièrement ouverte, de bout en bout, sur données réelles :
- modèle entraîné ;
- netlist câblée ;
- synthèse SKY130 ;
- placement OpenROAD ;
- simulation au niveau porte sur les **vraies images de test** ;
- puissance à partir du VCD.

Elle comprend la périphérie complète : tampon d'entrée, réseau, tête et argmax.

Ce qui manque dans l'existant :
- Fieldhouse 2026 utilise une activité **supposée** (0,7) ; les autres travaux comptent en portes,
  LUT ou NAND2-GE, sans énergie mesurée par simulation sur données réelles ni décomposition
  périphérique au niveau ASIC.
- Revendication défendable : « à notre connaissance, première décomposition surface/énergie
  annotée par l'activité réelle, sur un PDK ouvert, d'une inférence DLGN complète ».
- Formulation prudente, à revérifier avant soumission.

Preuves : `AUDIT.md` H1–H3 (reproductibles). À compléter :
- parasites routés (SPEF) ;
- simulation annotée en délais (glitches) ;
- tampon d'entrée avec horloge déclenchée par mot ;
- modèles entraînés au niveau de l'état de l'art.

### C-B — Quantification ASIC de la domination périphérique (confirmation, pas découverte)

Mesuré au niveau ASIC, ce qui avait été vu sur FPGA :
- tête ≈ 5x la surface des portes (60 k µm² pour 2000 votants) ;
- tampon d'entrée ≈ 60 % de la surface, 84 % de l'énergie sans horloge déclenchée.

Valeur : chiffres ASIC et énergie, là où la littérature donne des LUT. Présenté comme
**confirmation quantitative**.

### C-C — Résultats négatifs avec références équitables

| résultat | apport |
|---|---|
| pénaliser la longueur de fil à l'entraînement **n'a pas** raccourci les fils après placement libre (2000 portes) | réfute une intuition naturelle (OP4 de la revue) à petite échelle |
| réduire les votants par classe coûte 7 à 11 points | borne pratique pour réduire la tête |
| MAJ3 n'est pas meilleur que LUT2 par µm² | — |
| à 2000 portes, 1 seuil ≈ 3 seuils sur MNIST / Fashion-MNIST | cohérent avec LILogicNet sur CIFAR, étendu |
| arrêt certifié de l'acquisition sans gain face à un réseau à 1 seuil (à cette échelle) | évite une fausse piste |


### C-D — Caractérisation silicium indépendante (à venir, la plus forte)

- Mesure d'un DLGN fabriqué : précision sur puce, f_max, énergie dynamique différentielle.
- Comparaison avec la prédiction de notre chaîne pour **le même** design. C'est la validation de la
  méthode C-A par le silicium.
- Aucune mesure publiée à ce jour (`SILICON_STATUS.md`).
- Course possible avec les concepteurs : la collaboration est préférable à la compétition.

### C-E — Équivalence formelle modèle/matériel (atout transversal)

Un DLGN est une fonction booléenne pure. On peut **prouver** (vérification d'équivalence
SAT/ABC) que la netlist optimisée calcule exactement le modèle entraîné, sur toutes les entrées.

- Rare pour un accélérateur neuronal.
- Cohérent avec notre savoir-faire : riscv-formal et preuve d'équivalence SV/VHDL déjà faites
  sur le cœur.
- Contribution d'ingénierie de vérification, pas d'architecture.

## 3. Recommandation

- **Article 1 (court terme)** : C-A + C-B + C-C, avec C-E en section « vérification ».
  Pas de revendication d'architecture nouvelle.
- **Article 2 (après silicium)** : C-D, mesure contre prédiction.
- Candidats C1 (décision analogique certifiée) et C2 (arrêt certifié de l'acquisition) : publiés
  comme pistes explorées. C1 ne repose que sur des estimations ; C2 n'apporte pas de gain à cette échelle.
