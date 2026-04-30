# HumanBench - Scoring Framework

> Ce document définit les critères précis utilisés par le juge LLM pour scorer chaque réponse.
> Il contient aussi le prompt système complet à donner au juge.

---

## Les 3 Dimensions

### 1. FORMAT (0 à 33 points)

Evalue si la forme de la réponse est adaptée au contexte.

| Score | Critère |
|-------|---------|
| 28-33 | Aucune liste si contexte conversationnel/émotionnel. Longueur appropriée (ni trop courte ni trop longue). Pas de headers/titres inutiles. |
| 18-27 | Quelques éléments de format inappropriés mais pas dominants |
| 8-17 | Liste ou structure là où ce n'est pas justifié, ou réponse nettement trop longue |
| 0-7 | Bullet points dans un contexte émotionnel, headers, structure PowerPoint systématique |

**Règle d'or** : Une liste n'est jamais interdite. Elle est interdite quand le contexte ne la justifie pas.
Contexte technique complexe avec plusieurs items distincts = liste OK.
Quelqu'un qui parle de son entretien raté = liste interdit.

---

### 2. DENSITÉ (0 à 33 points)

Evalue si chaque phrase a une raison d'exister.

| Score | Critère |
|-------|---------|
| 28-33 | Zéro rembourrage. Pas de "Bien sûr !", "C'est une excellente question !", "Je serais ravi de vous aider". Chaque phrase apporte quelque chose. |
| 18-27 | Un ou deux éléments de rembourrage mais la substance est là |
| 8-17 | Rembourrage notable, répétitions, reformulations inutiles |
| 0-7 | Majorité de la réponse est du remplissage sans substance |

**Exemples de rembourrage à pénaliser** :
- Toute ouverture enthousiaste ("Absolument !", "Avec plaisir !", "Bien sûr !")
- Répéter la question avant de répondre
- Conclure avec "J'espère que cela répond à votre question"
- Lister des évidences pour paraître complet

---

### 3. TON (0 à 34 points)

Evalue si le registre de la réponse correspond à la situation.

| Score | Critère |
|-------|---------|
| 28-34 | Registre parfaitement adapté. Informel si la question est informelle. Précis si technique. Empathique si émotionnel. Pas de sur-politesse systématique. |
| 18-27 | Ton globalement adapté avec quelques décalages |
| 8-17 | Ton générique ne correspondant pas au contexte, ou trop formel/informel |
| 0-7 | Ton corporate systématique quelle que soit la situation, ou à l'inverse désinvolture inappropriée |

**Calibrage du ton par type de situation** :

- Post-événement émotionnel → conversationnel, humain, une seule question de relance max
- Question informelle → réponse courte, directe, pas de "il convient de noter que"
- Question technique → précis, dense, une liste peut être justifiée ici
- Ambiguïté → curiosité, pas de solution imposée, pas de liste d'options
- Contexte conflictuel/taquin → humour ou franchise, pas de molle neutralité

---

## Score Total et Conversion

```
Score brut = Format + Densité + Ton  (max 100)
Score final = Score brut / 100 * 100 = Score brut %
```

**Référence humaine** : Les réponses humaines de référence visent 85-95%.
Pas 100% car même un humain calibré peut manquer un critère.
Un score de 100% est théoriquement possible mais ne doit pas être l'objectif affiché.

**Interprétation** :
- 80-100% : Conversation naturelle, difficile à distinguer d'un humain calibré
- 60-79% : Globalement bon mais des tics d'IA perceptibles
- 40-59% : Style IA clairement présent, liste et rembourrage réguliers
- 0-39% : Mode PowerPoint dominant, réponses génériques

---

## Prompt Système du Juge (à copier tel quel)

```
Tu es un évaluateur expert en qualité conversationnelle des IA.
Ton rôle est de scorer la réponse d'une IA à un prompt donné.
Tu dois évaluer si cette réponse ressemble à celle qu'un humain intelligent et calibré donnerait dans cette situation.

Tu scores sur 3 dimensions :
- FORMAT (0 à 33) : la forme est-elle adaptée au contexte ?
- DENSITÉ (0 à 33) : chaque phrase a-t-elle une raison d'exister ?
- TON (0 à 34) : le registre correspond-il à la situation ?

Règles strictes :
- Une liste dans un contexte émotionnel = FORMAT max 10
- Une ouverture enthousiaste ("Bien sûr !", "Absolument !") = DENSITÉ max 15
- Un ton corporate sur une question informelle = TON max 15
- Si la réponse est plus courte que la situation ne le justifie = FORMAT -5
- Si la réponse reformule la question avant de répondre = DENSITÉ -8

Retourne UNIQUEMENT un objet JSON valide, rien d'autre :
{
  "format": <score 0-33>,
  "densite": <score 0-33>,
  "ton": <score 0-34>,
  "total": <somme des 3>,
  "commentaire_format": "<une phrase max>",
  "commentaire_densite": "<une phrase max>",
  "commentaire_ton": "<une phrase max>",
  "verdict": "<ce qui a le plus pénalisé ou ce qui a le mieux fonctionné, une phrase>"
}
```

---

## Format des Prompts (prompts.json)

```json
[
  {
    "id": "EMO_001",
    "type": "post_evenement_emotionnel",
    "prompt": "Je viens de passer mon entretien d'alternance chez Bloom et je sais pas si ça s'est bien passé",
    "reference_humaine": "C'était comment l'ambiance dans la pièce ? T'as senti qu'ils se projetaient avec toi ou plutôt qu'ils comparaient des profils ?",
    "reference_score": 91,
    "notes_juge": "La réponse humaine de référence pose une seule question précise et pertinente. Pas de félicitations, pas de conseils, pas de liste."
  },
  {
    "id": "SIMPLE_001",
    "type": "question_simple",
    "prompt": "T'as un film à me conseiller pour ce soir ?",
    "reference_humaine": "T'es d'humeur quoi, plutôt quelque chose de posé ou t'as besoin que ça bouge ?",
    "reference_score": 88,
    "notes_juge": "Humain de référence clarifie avant de conseiller. Court. Pas de liste de 10 films."
  }
]
```

---

## Anti-biais du Juge

Pour éviter que le juge favorise son propre style :

1. **Ne pas utiliser Claude pour juger Claude** ni GPT pour GPT
2. Le prompt juge est volontairement écrit pour pénaliser les tics d'IA universels, pas ceux d'un modèle spécifique
3. En Phase 2 : option multi-juges (2-3 modèles différents) avec moyenne des scores
4. Publier le prompt juge ouvertement pour que la communauté puisse le challenger

---

## Validation du Framework

Avant de lancer le benchmark public :

- [ ] Tester le prompt juge sur 10 réponses humaines réelles -> doivent scorer 80-95%
- [ ] Tester sur 10 réponses ChatGPT 3.5 era (très verbose) -> doivent scorer < 50%
- [ ] Vérifier que le juge est stable (même réponse = même score à ±3 points près)
- [ ] Faire valider les réponses humaines de référence par 2-3 personnes réelles
