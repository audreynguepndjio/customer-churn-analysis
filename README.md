# Customer Churn Analysis — Projet Machine Learning

## 1. Présentation du projet

Ce projet a pour objectif d’analyser le churn client d’une entreprise télécom et de construire un modèle de Machine Learning capable d’identifier les clients susceptibles de quitter l’entreprise.

Le churn correspond au départ ou à la résiliation d’un client. Dans un contexte business, détecter les clients à risque permet de mettre en place des actions de rétention ciblées.

---

## 2. Objectif métier

L’objectif principal est de répondre à la problématique suivante :

> Quels facteurs influencent le départ des clients et comment prédire les clients à risque de churn ?

Ce projet permet d’aider une entreprise à mieux comprendre les comportements de ses clients et à prioriser ses actions de fidélisation.

---

## 3. Questions business

Les principales questions traitées sont :

1. Quelle est la proportion de clients qui quittent l’entreprise ?
2. Le type de contrat influence-t-il le churn ?
3. L’ancienneté du client joue-t-elle un rôle dans le churn ?
4. Les montants mensuels élevés augmentent-ils le risque de départ ?
5. Certains services sont-ils associés à une meilleure rétention ?
6. Quel modèle de Machine Learning prédit le mieux le churn ?
7. Quelles recommandations business peut-on proposer ?

---

## 4. Dataset utilisé

Le dataset utilisé est **Telco Customer Churn**.

Il contient des informations sur les clients d’une entreprise télécom :

- informations démographiques ;
- type de contrat ;
- services souscrits ;
- ancienneté client ;
- montant mensuel ;
- montant total facturé ;
- statut de churn.

La variable cible est :

```text
Churn
```

Elle indique si le client a quitté l’entreprise ou non.

---

## 5. Outils utilisés

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib
- VS Code

---

## 6. Structure du projet

```text
customer-churn-analysis/
│
├── data/
│   ├── customer_churn_raw.csv
│   └── customer_churn_clean.csv
│
├── notebooks/
│   └── 01_customer_churn_analysis.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── churn_distribution.png
│   │   ├── churn_by_contract.png
│   │   ├── churn_by_tenure.png
│   │   ├── monthly_charges_by_churn.png
│   │   ├── model_comparison_f1_score.png
│   │   └── random_forest_feature_importance.png
│   │
│   └── results/
│       ├── model_comparison.csv
│       └── feature_importance.csv
│
├── models/
│   └── churn_random_forest_model.pkl
│
├── requirements.txt
└── README.md
```

---

## 7. Étapes du projet

### 7.1 Chargement des données

Les données ont été chargées avec Pandas depuis un fichier CSV.

```python
df = pd.read_csv("../data/customer_churn_raw.csv")
```

---

### 7.2 Compréhension du dataset

Le dataset initial contient :

- 7 043 lignes ;
- 21 colonnes ;
- une variable cible : `Churn`.

La majorité des variables sont catégorielles, ce qui nécessite un encodage avant la modélisation.

---

### 7.3 Nettoyage des données

La colonne `TotalCharges` était initialement reconnue comme une variable texte alors qu’elle représente un montant numérique.

Elle a donc été convertie en variable numérique :

```python
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
```

Après conversion, 11 valeurs manquantes ont été détectées puis supprimées, car elles représentaient une très faible part du dataset.

La colonne `customerID` a également été supprimée, car elle correspond à un identifiant unique et n’apporte pas d’information utile pour la prédiction.

Après nettoyage, le dataset contient :

- 7 032 lignes ;
- 20 colonnes.

---

## 8. Analyse exploratoire

### 8.1 Répartition du churn

Le churn concerne environ 26,5 % des clients.

Cela signifie que le dataset est modérément déséquilibré :

- environ 73,5 % des clients ne churnent pas ;
- environ 26,5 % des clients churnent.

Cette information est importante pour l’évaluation des modèles, car l’accuracy seule ne suffit pas.

---

### 8.2 Type de contrat

L’analyse montre que les clients avec un contrat `Month-to-month` churnent beaucoup plus que ceux ayant un contrat d’un an ou deux ans.

Cela suggère que l’engagement contractuel joue un rôle important dans la fidélisation.

---

### 8.3 Ancienneté client

Les clients récents sont plus exposés au churn.

L’ancienneté (`tenure`) apparaît comme un facteur clé : plus un client reste longtemps, plus son risque de churn diminue.

---

### 8.4 Montant mensuel

Les clients qui churnent ont tendance à avoir des `MonthlyCharges` plus élevés.

Cela peut indiquer que le prix ou la perception de valeur du service influence le départ des clients.

---

### 8.5 Services et moyens de paiement

Plusieurs variables semblent liées au churn :

- les clients utilisant `Fiber optic` churnent davantage ;
- les clients payant par `Electronic check` présentent un taux de churn plus élevé ;
- les clients disposant de `OnlineSecurity` churnent moins ;
- les clients disposant de `TechSupport` churnent moins.

---

## 9. Prétraitement pour le Machine Learning

Les étapes de prétraitement ont été :

1. encodage de la variable cible `Churn` :
   - `No` → 0 ;
   - `Yes` → 1.

2. séparation des variables explicatives et de la cible :

```python
X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]
```

3. encodage des variables catégorielles avec `pd.get_dummies()` :

```python
X_encoded = pd.get_dummies(X, drop_first=True)
```

4. séparation train/test avec stratification :

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

5. standardisation des données pour la régression logistique.

Après encodage, le dataset contient 30 variables explicatives.

---

## 10. Modèles testés

Trois modèles de classification ont été entraînés :

- Régression logistique ;
- Arbre de décision ;
- Random Forest.

Les modèles ont été comparés avec les métriques suivantes :

- Accuracy ;
- Precision ;
- Recall ;
- F1-score ;
- Matrice de confusion.

Dans un contexte de churn, le **recall** est particulièrement important, car l’objectif est de détecter un maximum de clients réellement à risque.

---

## 11. Résultats des modèles

Le modèle **Random Forest** a obtenu le meilleur F1-score parmi les modèles testés.

La matrice de confusion du Random Forest montre :

| | Prédit No Churn | Prédit Churn |
|---|---:|---:|
| Réel No Churn | 764 | 269 |
| Réel Churn | 82 | 292 |

Le modèle détecte 292 clients churn sur 374 dans l’échantillon de test, soit un recall d’environ 78 % pour la classe churn.

Cela signifie que le modèle identifie une grande partie des clients à risque, même s’il génère aussi des faux positifs.

---

## 12. Importance des variables

Le modèle Random Forest identifie les variables les plus importantes dans la prédiction du churn.

Les variables les plus influentes sont notamment :

- `tenure` ;
- `TotalCharges` ;
- `Contract_Two year` ;
- `MonthlyCharges` ;
- `InternetService_Fiber optic` ;
- `PaymentMethod_Electronic check` ;
- `Contract_One year` ;
- `OnlineSecurity_Yes` ;
- `TechSupport_Yes`.

Ces résultats sont cohérents avec l’analyse exploratoire.

---

## 13. Recommandations business

À partir de l’analyse et des résultats du modèle, plusieurs recommandations peuvent être proposées.

### 13.1 Renforcer l’accompagnement des nouveaux clients

L’ancienneté est la variable la plus importante. Les clients récents présentent un risque plus élevé de churn.

Il serait pertinent de renforcer l’onboarding client pendant les premiers mois avec un suivi personnalisé, une assistance proactive et des offres de fidélisation.

### 13.2 Encourager les contrats longue durée

Les clients avec des contrats d’un ou deux ans churnent beaucoup moins que les clients avec un contrat mensuel.

L’entreprise pourrait proposer des avantages pour inciter les clients à passer vers des contrats longue durée.

### 13.3 Surveiller les clients avec charges mensuelles élevées

Les clients ayant des montants mensuels élevés présentent un risque de churn plus important.

Une analyse complémentaire pourrait permettre d’identifier les clients ayant un coût élevé mais une faible perception de valeur.

### 13.4 Analyser l’offre Fiber optic

Les clients utilisant l’offre `Fiber optic` churnent davantage.

Il serait utile d’analyser la satisfaction, le prix, la qualité du service ou la concurrence sur cette offre.

### 13.5 Réduire le risque lié au paiement par Electronic check

Le paiement par `Electronic check` est associé à un churn plus élevé.

L’entreprise pourrait encourager les paiements automatiques, qui semblent associés à une meilleure fidélisation.

### 13.6 Promouvoir les services additionnels

Les services `OnlineSecurity` et `TechSupport` semblent contribuer à réduire le churn.

Ces services peuvent être proposés aux clients à risque comme leviers de rétention.

### 13.7 Mettre en place un score de risque de churn

Le modèle peut être utilisé pour attribuer un score de risque à chaque client.

Les équipes marketing ou relation client pourraient ensuite prioriser les actions de rétention sur les clients les plus exposés.

---

## 14. Sauvegarde du modèle

Le modèle final a été sauvegardé avec Joblib :

```python
joblib.dump(model_artifact, "../models/churn_random_forest_model.pkl")
```

Le fichier généré est disponible dans le dossier :

```text
models/
```

---

## 15. Comment exécuter le projet

### 15.1 Installer les dépendances

```bash
pip install -r requirements.txt
```

### 15.2 Ouvrir le notebook

```bash
jupyter notebook notebooks/01_customer_churn_analysis.ipynb
```

---

## 16. Compétences démontrées

Ce projet met en évidence plusieurs compétences :

- analyse et compréhension d’un besoin métier ;
- nettoyage et préparation de données ;
- analyse exploratoire ;
- visualisation avec Matplotlib et Seaborn ;
- encodage de variables catégorielles ;
- séparation train/test ;
- standardisation ;
- entraînement de modèles de Machine Learning ;
- évaluation de modèles avec des métriques adaptées ;
- interprétation des variables importantes ;
- formulation de recommandations business ;
- sauvegarde d’un modèle ML.

---

## 17. Conclusion

Ce projet montre comment l’analyse de données et le Machine Learning peuvent aider une entreprise à mieux comprendre les causes du churn et à cibler ses actions de fidélisation.

L’analyse a permis d’identifier plusieurs facteurs de risque : faible ancienneté, contrat mensuel, charges mensuelles élevées, paiement par Electronic check, absence de sécurité en ligne et absence de support technique.

Le modèle Random Forest a été retenu comme modèle final grâce à son meilleur F1-score et à sa capacité à détecter une grande partie des clients à risque.

Ce projet constitue une base solide pour un portfolio Data Analyst / Machine Learning junior.