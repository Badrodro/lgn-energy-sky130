# Recherche d'une architecture de calcul nouvelle — livrable initial (phases 1 à 4 + protocole)

Date : 23/10/2026. Les registres de lecture détaillés (A à D) ne sont pas publiés.

Étiquettes : **[P]** prouvé (mathématique/physique), **[E]** empirique (source), **[A]** supposé,
**[M]** mesuré dans ce projet (sky130, voir `review/E_measured_sky130.md`).

---

## 1. Méthode et couverture (phase 1)

- Quatre revues parallèles, de la revue de synthèse vers les travaux fondateurs puis vers les
  travaux récents qui les citent, brevets cherchés séparément :
  - **A** accélérateurs neuronaux et compression/sparsité : 81 sources + 11 brevets.
  - **B** NMC/PIM/CIM, calcul associatif, arithmétique approximative et systèmes de numération :
    65 sources + 21 brevets.
  - **C** réseaux logiques/LUT/sans poids, méthodes de co-conception et modèles de coût :
    92 sources (dont 1 brevet lu en partie).
  - **D** RISC-V, interconnexions, puces d'inférence toujours actives (MLPerf Tiny) :
    52 sources + 6 brevets.
  - Environ 95 sources lues au-delà du résumé.
- **Limites de couverture (à ne pas oublier)** :
  - Google Patents, USPTO, Espacenet, IEEE Xplore et ACM DL non consultés. Les
    brevets ne sont connus que par les extraits du moteur de recherche ; les revendications n'ont
    pas été lues, sauf US 12 688 430 B1 (revendications 1 à 18 sur 49).
  - Semantic Scholar a fortement limité les requêtes, donc les graphes de citations sont partiels.
  - Plusieurs chiffres ISSCC/JSSC payants viennent de métadonnées (marqués M/UNVERIFIED).
  - **Ce n'est pas une revue exhaustive.** Une recherche d'antériorité professionnelle reste
    indispensable avant tout dépôt.

## 2. État de l'art — points saillants (détails et sources dans les registres)

### Accélérateurs, LLM, compression (A)
- Le décodage des LLM est limité par la bande passante [P, roofline] : environ 2 op/octet à batch 1
  en INT8. L'écart calcul/bande passante se creuse (3,0x contre 1,6x tous les 2 ans) [E].
- Mouvement de données >> calcul : DRAM 640 pJ contre 0,1 pJ pour une addition (45 nm) [E] ;
  sky130 : lecture SRAM 32 bits = 13,8 pJ contre ~13 fJ par transition d'additionneur [M].
- Les gains BitNet (71x) sont un modèle d'énergie arithmétique, pas du silicium [E].
- N:M ne se combine pas avec ≤ 2 bits (métadonnées = 100 %) [P, arithmétique].
- La sparsité d'activation s'efface avec le batch [E].

### Calcul en mémoire, associatif, approximatif (B)
- **Mur de l'ADC** [P] : 6 dB/bit ; énergie ~4^B limitée par le bruit ; l'analogique ne gagne
  qu'en dessous de ~6-8 bits.
- Écart crête/système [E] :
  - CIM SRAM 17x plus efficace par banc, mais ~3x au niveau processeur ;
  - HDC en mémoire : 334x sur le module, 6x de bout en bout.
- Les surcoûts cachés (ADC, calibration, logique numérique, hôte) dominent [E].
- Mitchell : erreur −11,1 % unilatérale, prouvée [P] ; stochastique : flux de 2^n pour n bits [P].

### Réseaux logiques et co-conception (C)
- MNIST : 98,45 % avec 8 000 portes, câblage appris [E]. CIFAR-10 : 86,3 % à 61 M portes [E].
- **La tête de sortie (GroupSum/popcount/argmax) coûte autant ou plus que le réseau** [E] :
  - 75 % des LUT (LILogicNet) ;
  - 59 k contre 61 k NAND2-GE (BitLogic) ;
  - ~5x la surface des portes [M].
- L'encodage d'entrée est un coût caché (jusqu'à 3,2x) [E]. Les fils et le routage limitent la
  taille : 384 k portes non routables en sky130 [E].
- **Aucun silicium mesuré de réseau logique différentiable** [E]. Le seul chiffre ASIC suppose une
  activité de 0,7 [E].
- Les bornes de mouvement de données (Hong–Kung) supposent les poids en mémoire. Aucune borne
  n'existe pour les paramètres câblés [P/A].

### RISC-V et toujours actif (D)
- La puissance moyenne d'un système toujours actif est fixée par le repos, les E/S, la rétention
  mémoire et le frontal analogique, pas par le calcul d'inférence [E] :
  - rétention SRAM 60 à 150 nW/ko ;
  - frontal ~70 % (Vocell) ;
  - E/S 230 µW (Xylo).
- MLPerf Tiny KWS : 18,6 µJ (GAP9) à 245 µJ (STM32U385) ; le prétraitement et le repos sont exclus
  par construction [E].
- Les cœurs programmables dépensent l'essentiel de leur énergie en contrôle : 28 pJ utiles sur
  317 pJ dans la mesure de motivation de Snitch [E].

## 3. Problèmes fondamentaux extraits (phase 2)

| id | problème | énoncé | statut |
|---|---|---|---|
| F1 | mouvement de données | tout bit de paramètre non résident au point de calcul se déplace au moins une fois par inférence ; seules échappatoires : résidence (CIM, câblage) ou moins de bits | [P] borne, [E][M] chiffres |
| F2 | réduction | somme exacte de N termes de b bits : Θ(N·b) cellules (numérique) ; analogique : O(1) composant par terme, mais lecture de précision log2N + b bits, ADC ~2^bits (flash) ou bits étapes (SAR), bruit σ√N | [P] |
| F3 | **la décision demande moins que la valeur** | argmax_k S_k, signe(S − θ) ou top-1 ne dépendent que de l'ordre : un comparateur (1 bit) suffit quand la marge dépasse l'erreur | [P] |
| F4 | taxe de programmabilité | routage/stockage reconfigurables : ~35x surface, ~14x puissance (FPGA/ASIC) | [E] |
| F5 | fuite à faible rapport cyclique | P_moy = P_fuite + P_rétention + P_E/S + f·E_inf ; en dessous de ~1 inf/s, les trois premiers dominent | [P] + [E] |
| F6 | la périphérie domine | entrée, sortie, E/S, contrôle >> cœur de calcul (réseaux logiques, CIM, MCU) | [E][M] |
| F7 | certification serrée seulement sur incréments discrets bornés | bornes au pire cas d'une somme réelle en grande dimension lâches d'un facteur ~√N ; bornes sur des comptes/bits serrées | [P] (argument), [M] (DTN, arrêt précoce) |

Croyances courantes identifiées comme **non démontrées** [A] :
- « les réseaux logiques sont des ordres de grandeur plus efficaces » (estimations FPGA,
  activités supposées) ;
- « l'analogique gagne toujours en énergie » (faux au niveau système et au-delà de ~6-8 bits) ;
- « TOPS/W prédit l'autonomie » (faux en toujours actif, F5).

## 4. Motif transversal et formulation mathématique (phase 3)

### 4.1 Le motif « sommer puis décider »
Dans trois domaines qui ne communiquent pas, on calcule une grandeur de haute précision pour n'en
extraire qu'une **décision** :
- tête des réseaux logiques : popcount puis argmax (C, F6) ;
- CIM analogique : somme de colonne, ADC puis signe/ReLU/argmax (B, F2) ;
- capteurs toujours actifs : signal complet puis décision (D, F5).

Le coût est dominé par la précision de la valeur (popcount Θ(N), ADC ~4^B), alors que la décision
n'exige que l'ordre (F3).

### 4.2 Formulation
- Votes binaires v_i ∈ {0,1}, groupes G_k (|G_k| = m), scores S_k = Σ_{i∈G_k} v_i, décision exacte
  D = plus petit indice de argmax_k S_k.
- Mesure analogique Ŝ_k = S_k + e_k. Pour une comparaison de paires, l'erreur e_ab sur Ŝ_a − Ŝ_b
  vérifie |e_ab| ≤ δ avec probabilité ≥ 1 − ε (décalage du comparateur, désappariement des
  capacités, bruit kT/C).

**Lemme de certification [P].** Si Ŝ_a − Ŝ_b > δ, alors S_a > S_b avec probabilité ≥ 1 − ε.
Une classe a est un **gagnant certifié** si cela vaut pour tout b ≠ a. On a alors D = a avec
probabilité ≥ 1 − (K−1)ε (borne de l'union).

**Procédure exacte.**
1. Comparateurs à zone morte : pour chaque paire, trois issues (a ≻ b, b ≻ a, incertain).
2. S'il existe un gagnant certifié, le sortir.
3. Sinon, recompter **exactement** (numérique séquentiel) les seules classes non dominées de
   manière certifiée, puis appliquer l'argmax exact avec la même règle d'égalité.

La sortie est **identique au modèle numérique de référence**, à ε près (ε réglable, par exemple
δ = 6σ donne ε ≈ 10⁻⁹ par comparaison).

**Coût attendu.** E = E_analog + P_f·E_repli, avec P_f = P(marge(S) ≤ δ + erreur) ≈ F_marge(δ).
- Mesure [M] sur un réseau logique MNIST (200 votants/classe, 92,7 %) : P(marge ≤ 3 votes) =
  3,0 %, ≤ 5 = 4,6 %, ≤ 12 = 10,4 %.

**Dimensionnement optimal (résultat analytique, à valider).**
- Loi de Pelgrom : σ_offset ∝ 1/√(WL), donc l'énergie du comparateur vaut E_c ≈ c/σ².
- Avec δ = kσ et une densité de marge f₀ près de 0, P_f ≈ f₀·k·σ.
- D'où E(σ) = c/σ² + f₀·k·σ·E_repli, minimale en **σ\* = (2c / (f₀·k·E_repli))^(1/3)**.

## 5. Candidats (phase 4)

### C1 — Décision analogique certifiée par marge avec repli numérique exact (« MCD »)

- **Contribution exacte.** Mécanisme général pour tout étage « sommer puis décider » (tête argmax,
  activation signe/seuil, top-1) :
  - accumulation en charge (une capacité unité par vote, pas d'additionneur) ;
  - comparateurs à zone morte réglée au-delà de la borne d'erreur analogique ;
  - recomptage numérique séquentiel **uniquement** des classes non certifiées.
  - Résultat : décision bit-exacte par rapport au modèle numérique, à ε contrôlé près.
  - Avec le dimensionnement optimal σ\*.
- **Avantage supposé** (à démontrer, estimations [A]) :
  - tête 6 à 12x plus petite que le popcount numérique (60 215 µm² mesurés pour 2 000 votants) ;
  - ~10x moins d'énergie ;
  - exactitude conservée, donc vérifiable formellement (argument commercial sûreté/certification).
- **Contreparties.**
  - Conception analogique : calibration, PVT, injection de charge, couplage.
  - Latence variable (repli dans ~3 à 5 % des cas).
  - Test de production.
- **Résultats qui l'invalideraient.**
  - P_f > ~20 % sur des modèles réalistes (KWS, F-MNIST, CIFAR).
  - Décalage des comparateurs sky130 imposant δ > ~10 votes.
  - Une tête **numérique séquentielle** (compteurs + arrêt certifié) atteignant la même surface
    et la même énergie sans analogique.
  - Erreurs systématiques (couplage) non bornables.
- **Cinq travaux les plus proches et différences.**
  1. Bankman, Murmann et al., JSSC 2019 : BNN à neurones à capacités commutées et comparateur.
     Accumulation en charge plus comparateur, mais **sans certification ni repli** ; précision
     limitée par l'analogique.
  2. Zheng et al. (Duke), DAC 2024 : macro CIM hybride analogique-numérique « sans perte ». Le
     partage se fait selon la **taille** des valeurs (petites en analogique, grandes en
     numérique) pour obtenir des valeurs exactes, et non selon la **marge de décision**.
  3. Demirkiran et al., Nature Comms 2024 : réseaux analogiques tolérants aux fautes par RRNS.
     Codes redondants, coût en tableaux supplémentaires ; pas d'exploitation de la décision.
  4. Codes correcteurs analogiques (Roth et al.) : corrigent les erreurs de calcul en mémoire ;
     même distinction que ci-dessus.
  5. WTA analogique (Lazzaro et al. 1989 ; brevet US 5 059 814) : argmax analogique **sans
     garantie** ; échoue en présence de bruit pour N grand.
  - À vérifier aussi : l'option de rejet (Chow 1970) et les cascades de classifieurs. Elles sont
    approximatives (modèles différents), alors que C1 garde le même modèle et l'exactitude.
  - **Aucune source trouvée** sur la certification par marge avec recalcul numérique des seuls
    cas ambigus. C'est un signal, pas une preuve.

#### C1 — premiers résultats de la phase 5 (23/10/2026)

| mesure | valeur | statut |
|---|---|---|
| Marges (MNIST, 2000 votants) | P(marge ≤ 3) = 3,0 % ; ≤ 5 = 4,6 % | [M] — étape 1 franchie |
| Comparateur StrongARM sky130 (entrées 2/0,5 µm), Monte-Carlo tt_mm | σ_offset ≈ 3,7 mV (≈ 0,4 vote à 9 mV/vote), 65,6 fJ/décision | [M] partiel (10 tirages) — étape 2 franchie |
| Tête numérique R1 (arbre / Wallace), 2000 votants, VCD réel, sans fils | 61 969 / 60 111 µm² ; 71 / 68 pJ par décision ; exacte (199/199) | [M] |
| Désappariement MIM sky130 (modèle PDK) | σ(ΔC/C) = 2,8 % / √(aire µm²) | [M] (modèle PDK) |
| **Repli analogique exact** (erreur < 0,5 vote à 6σ) | capacités ≈ 34 µm² (≈ 68 fF) → ≈ 190 pJ/décision > numérique | [E, calcul] → **réfuté** (mur de précision, cohérent avec B) |
| Variante exacte avec repli numérique série | ≈ 2x surface, ≈ 5x énergie (estimation) | [E] — gain modeste |
| Variante approchée sans repli | ≈ 10x surface ; proche de Bankman 2019 ; perd l'exactitude | [E] — faible nouveauté |

**Verdict provisoire C1** : principe correct, mais plafonné par la physique (précision analogique
~11 bits requise pour l'égalité exacte sur 200 votes). Pas une rupture.

### C2 — Décision certifiée pendant l'acquisition (arrêt anticipé exact du capteur)

- **Contribution.** Les votes s'accumulent pendant que l'entrée arrive :
  - rampe d'ADC, bits de thermomètre par seuil croissant ;
  - ou lignes d'image / trames audio.
- La décision est certifiée dès que l'avance du meneur dépasse le **maximum** de votes encore
  possibles pour les autres. Ce sont des incréments bornés, donc des bornes serrées (F7).
- On arrête alors l'acquisition et la conversion, ce qui vise le frontal qui domine la puissance
  toujours active (D).
- **Proches.** Early-exit et réseaux anytime (approximatifs) ; décision anticipée dans les ADC SAR
  (redondance) ; CascadeLUT (ordonnancement de l'entrée).
- **Invalidation.** Fraction certifiée avant la fin de l'acquisition trop faible (< 20 % de gain
  moyen), ou bornes trop lâches pour les architectures utiles.
- **Statut.** Analyse « rampe » exécutée (simulation à 3 valeurs du réseau discrétisé, règle
  d'égalité de l'argmax respectée).

#### C2 — premiers résultats de la phase 5 (23/10/2026)

| modèle (MNIST, seuils 0,25/0,5/0,75) | certifié après seuil 1 | après seuil 2 | durée moyenne de rampe | statut |
|---|---|---|---|---|
| 2000 portes, 92,7 % | 64,2 % | 88,6 % | 49 % (≈ 2,0x) | [M] |
| 1000 portes, 90,5 % | 73,4 % | 92,4 % | 45 % (≈ 2,2x) | [M] |
| ordre SAR (0,5 d'abord), 2000 portes | 0 % | 100 % | — | [M] : ordre inadapté aux fonds noirs |

- L'ordre d'acquisition doit être optimisé pour les données (problème bien posé : calendrier de
  comparaisons maximisant la certification attendue).
- Antériorité vérifiée par moteur de recherche :
  - lecture conditionnelle par pixel (brevets Rambus, capteurs suréchantillonnés) ;
  - arrêt anticipé de SAR sur condition déterministe.
  - Rien trouvé sur l'arrêt de l'acquisition dès que la **décision globale** est certifiée par
    propagation 0/1/X dans le classifieur.
- À faire :
  - Fashion-MNIST ;
  - plus de seuils ;
  - coût matériel de l'évaluation à 3 valeurs (double rail) ;
  - modèle d'énergie du frontal.

### C3 — Modèle analytique du croisement spatial/temporel pour les réseaux logiques (OP6 de C)

- **Contribution.** Une formule du facteur de réutilisation optimal (portes physiques par porte
  logique) en fonction du débit cible, de l'énergie SRAM/registre et du fil, validée en sky130.
- Outil plus que brevet ; utile pour positionner C1/C2.

### Hors de portée ici (noté pour mémoire)
- CIM en résidus (RNS) avec ADC à compteur qui reboucle naturellement (B, H2).
- CIM sur mantisses MX avec exposant numérique (B, H1).
- Ces deux pistes demandent du silicium analogique avancé et des LLM : non vérifiables dans cet
  environnement.

### Déjà réfutés dans ce projet (mesures, voir `review/E_measured_sky130.md`)
- Arbres de dominance (DTN), max-plus, comparateurs appris en entrée, MAJ3 contre LUT2, moins de
  votants.
- Câblage pénalisé par la longueur (réfuté par placement réel).
- Câblage papillon fixe (×50 en portes).

## 6. Protocole pour départager (phase 5)

### Références équitables, toutes mesurées dans la même chaîne sky130
Synthèse → placement OpenROAD → simulation porte à porte sur données réelles → puissance VCD,
plus ngspice pour l'analogique.
- **R1** tête numérique parallèle (popcount en arbre + argmax).
- **R2** tête numérique **séquentielle** : compteurs, 32 votes/cycle, avec et sans arrêt certifié.
  C'est le concurrent le plus dangereux de C1.
- **R3** tête analogique **sans** certification (style Bankman/WTA) : mesure de la perte de
  précision.

### Étapes et critères (arrêt dès qu'un critère d'invalidation tombe)
1. **Marges** : distribution de la marge top-1/top-2 sur 3 tâches (MNIST, Fashion-MNIST, un KWS ou
   CIFAR-10 réduit) et 2 tailles de modèle → P_f(δ). *Échec si P_f(5 votes) > 20 %.*
2. **Comparateur** : StrongARM sky130 en ngspice, Monte-Carlo de désappariement → σ_offset, énergie
   par décision, délai → δ et σ\*. *Échec si δ > 10 votes à énergie raisonnable.*
3. **Capacités** : réseau de capacités unité MIM/MOM, désappariement, couplage, injection de charge
   (ngspice) → erreur bornée.
4. **Tête complète** : surface et énergie C1 contre R1, R2, R3 à exactitude égale (R1/R2 exacts,
   C1 exact à ε). *Échec si C1 n'est pas ≥ 3x meilleur que R2 en surface ou en énergie.*
5. **Système de bout en bout** : entrée en flux + réseau + tête, énergie par inférence et puissance
   moyenne à 1-10 Hz, y compris fuite et rétention.
6. **Antériorité** : recherche ciblée sur les revendications (conseil en PI) avant tout dépôt.

Séparer strictement dans les résultats : **démontré** (mesure/preuve), **estimé** (modèle),
**supposé**.
