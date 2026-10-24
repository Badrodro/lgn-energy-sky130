# Plan minimal de démonstrateur SKY130 crédible (23/10/2026)

Objectif : une preuve **vérifiable** (mesures sur silicium comparées à des prédictions publiées à
l'avance), pas une revendication d'innovation.

## Phase 0 — Prérequis côté simulation (sans budget, ~3-6 semaines)

1. **Modèles au niveau de l'état de l'art.** Connexions apprises style LILogicNet/Mommen, 200 époques,
   ≥ 3 graines. Cibles indicatives :
   - MNIST ≥ 98 % à ~8 k portes ;
   - entrées 16×16 pour limiter les E/S.
2. **Puissance routée.** Routage complet, extraction SPEF, puissance VCD sur parasites routés.
3. **Simulation annotée en délais (SDF).** Mesure de la part de glitchs, absente aujourd'hui (AUDIT
   H2).
4. **Tampon d'entrée** à horloge déclenchée par mot (corrige les 84 % mesurés).
5. **Équivalence formelle** netlist finale ≡ modèle entraîné (ABC `cec` / Yosys `equiv`) sur toutes
   les entrées.
6. **Prédictions figées et horodatées** dans le dépôt, avant toute mesure silicium :
   précision, f_max (STA), énergie par inférence, surface.

## Phase A — Caractériser une puce DLGN existante (≈ 300 € + instrument, 1-2 mois)

- **Puce** : Tiny Tapeout ttsky25a, projet n°72 `tt_um_rejunity_lgn_mnist` (Zioma & Armaitis,
  MNIST 16×16, ~10 k portes, open source).
  - Kit de développement Tiny Tapeout ~300 € (disponibilité à vérifier : affiché épuisé par moments).
  - Prise de contact avec les concepteurs **avant** publication : remerciements ou co-signature.
- **Prédiction** : reconstruire leur design depuis leur dépôt et l'analyser avec notre chaîne
  (surface, STA, énergie VCD). Les chiffres sont figés avant la mesure.
- **Mesures** :

| métrique | méthode | limite |
|---|---|---|
| exactitude fonctionnelle | 10 000 images de test envoyées par le RP2350 de la carte ; sorties comparées bit à bit au modèle de référence | débit des E/S (horloge projet 1 MHz nominale) |
| précision | sur le jeu de test complet | — |
| f_max | balayage d'horloge jusqu'aux premières erreurs ; comparaison à la STA | limité par les E/S et le multiplexeur TT |
| énergie dynamique / inférence | SMU ou ampèremètre de précision sur le rail cœur 1,8 V. Différence projet actif/inactif, à plusieurs fréquences : la pente ∝ f donne l'énergie dynamique | rail partagé par toute la puce : seule l'énergie **différentielle** est attribuable ; fuite non attribuable ; E/S 3,3 V non comptées |

- **Résultat attendu** : précision sur puce, f_max et énergie dynamique, avec l'erreur de prédiction
  de notre chaîne. C'est la validation de la méthode (C-D de `CONTRIBUTION.md`).

## Phase B — Notre propre puce sur navette (6-12 mois, budget à chiffrer)

- **Navette** : Tiny Tapeout (SKY130 ou IHP/GF selon calendrier), quelques tuiles. Ou ChipFoundry
  (harnais Caravel) si l'on veut un **domaine d'alimentation séparé**, donc une énergie plus propre
  (coût nettement plus élevé). Tarifs et calendriers à confirmer au moment du choix.
- **Contenu** : deux variantes du même réseau, pour mesurer **directement** l'effet de la
  périphérie.
  - V1 : périphérie classique (registre d'entrée parallèle, popcount complet).
  - V2 : périphérie optimisée (tampon à horloge déclenchée, tête réduite).
  - Option : interface avec le cœur RV32I vérifié formellement (déjà prêt).
- **Métriques imposées** :

| métrique | définition | référence |
|---|---|---|
| précision | jeu de test complet, sortie puce comparée bit à bit au modèle | modèle logiciel |
| surface | surface placée du macro (GDS), µm² | prédiction Phase 0 |
| timing | f_max mesurée contre STA (coin TT) | STA signée |
| énergie | dynamique différentielle par inférence, à 2-3 fréquences | prédiction VCD routée |
| fuite | mesurée seulement si domaine séparé | — |

- **Limites à déclarer** : un seul lot, peu d'échantillons, rail partagé (Tiny Tapeout), E/S lentes,
  PDK à but éducatif.

## Ce que ce démonstrateur NE prouve pas

- Aucune supériorité sur des NPU commerciaux (nœuds, précisions et tâches différents).
- Aucun résultat au-delà de MNIST / Fashion-MNIST.
