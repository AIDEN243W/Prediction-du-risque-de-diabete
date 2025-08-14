# Prédiction du risque de diabète

Une Approche d'Apprentissage Automatique (Machine Learning) Utilisant le Jeu de Données Pima Indians Diabetes

Ce dépôt présente un modèle d'apprentissage automatique entraîné sur le jeu de données Pima Indians Diabetes, qui contient diverses caractéristiques médicales de patientes. L'objectif de ce projet est de prédire avec précision la probabilité de diabète en utilisant des attributs tels que le taux de glucose, la pression artérielle, l'indice de masse corporelle (IMC), l'âge, et plus encore. Le modèle utilise des techniques établies d'apprentissage automatique pour analyser ces caractéristiques et fournir des insights sur le risque de diabète, ce qui en fait un outil précieux pour les professionnels de la santé et les chercheurs cherchant à comprendre la prévalence du diabète.  

## Table des Matières  

- [Aperçu du Projet](#aperçu-du-projet)  
- [Description du Jeu de Données](#description-du-jeu-de-données)  
- [Instructions d'Installation](#instructions-dinstallation)  
- [Modèle et Techniques](#modèle-et-techniques)      
- [Résultats et Performances](#résultats-et-performances)
- [Améliorations](#améliorations)  
    - [Méthode d'Imputation KNN pour les Valeurs Manquantes](#méthode-dimputation-knn-pour-les-valeurs-manquantes)
    - [Méthode PCA pour la Sélection des Caractéristiques](#méthode-pca-pour-la-sélection-des-caractéristiques) 
    - [Nouvelles Méthodes de Validation](#nouvelles-méthodes-de-validation)
    - [Modèle Random Forest](#modèle-random-forest)
    - [Optimisation du Random Forest](#optimisation-du-random-forest)
- [Instructions d'Utilisation](#instructions-dutilisation)  
- [Travaux Futurs](#travaux-futurs)

## Aperçu du Projet

Ce projet vise à construire un modèle prédictif pour le diabète en utilisant le jeu de données Pima Indians Diabetes. Le modèle aide à identifier les individus à risque de diabète en se basant sur des attributs médicaux tels que l'âge, l'IMC, les niveaux d'insuline, et plus encore.

## Description du Jeu de Données

Le jeu de données Pima Indians Diabetes provient du National Institute of Diabetes and Digestive and Kidney Diseases. Il inclut des données sur 768 patientes d'origine amérindienne Pima, avec les attributs suivants :
- Grossesses
- Glucose
- Pression Artérielle
- Épaisseur de la Peau
- Insuline
- IMC
- Fonction Pedigree Diabète
- Âge
- Résultat (0 ou 1, indiquant l'absence ou la présence de diabète)

## Instructions d'Installation  
Pour exécuter ce projet, assurez-vous d'avoir Python installé ainsi que Jupyter Notebook. Vous aurez également besoin des bibliothèques suivantes :  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- imblearn

Pour installer ces bibliothèques, vous pouvez utiliser pip :  
```bash  
pip install pandas numpy scikit-learn matplotlib seaborn imblearn
```

## Modèle et Techniques  

Ce projet utilise un classifieur Naive Bayes pour prédire le diabète. Les étapes clés incluent :  
- Prétraitement des données : Gestion des valeurs manquantes, détection des valeurs aberrantes, normalisation et équilibrage des données, sélection des caractéristiques pour identifier les attributs les plus significatifs, et division des données en ensembles d'entraînement et de test.  
- Entraînement du modèle : Utilisation d'un classifieur Naive Bayes pour entraîner le modèle sur l'ensemble d'entraînement.  
- Évaluation du modèle : Évaluation des performances du modèle en utilisant l'exactitude, la précision, le rappel, le F1-score, le rapport de classification, la matrice de confusion et le score AUC.  

## Résultats et Performances

Le modèle a atteint une exactitude de 78% sur l'ensemble de test. Les métriques de performance clés incluent :
- Précision : 0.63
- Rappel : 0.56
- F1-score : 0.59
- Score AUC : 0.71

Des métriques de performance détaillées et des visualisations sont disponibles dans la section des résultats du dépôt.

## Améliorations  

Dans ce projet, plusieurs améliorations ont été apportées pour optimiser le prétraitement des données et les performances du modèle. 

1. **Imputation KNN pour les Valeurs Manquantes.**
2. **Méthode PCA pour la Sélection des Caractéristiques.**
3. **Nouvelles Méthodes de Validation** 
4. **Ajout du modèle Random Forest**
5. **Optimisation du Random Forest**

### Méthode d'Imputation KNN pour les Valeurs Manquantes  

Dans cette mise à jour, j'ai ajouté une nouvelle méthode pour gérer les valeurs manquantes en utilisant l'imputation KNN. Cette méthode améliore l'étape de prétraitement des données en fournissant une manière plus robuste d'imputer les valeurs manquantes basée sur les plus proches voisins. Les avantages de l'imputation KNN sont :  
- **Préservation des Relations** : KNN prend en compte la similarité des points de données, ce qui aide à maintenir les relations entre les caractéristiques lors de l'imputation des valeurs manquantes.  
- **Amélioration des Performances du Modèle** : Une gestion appropriée des données manquantes peut conduire à une meilleure exactitude et fiabilité du modèle.

### Méthode PCA pour la Sélection des Caractéristiques

Un aspect clé de cette analyse est la sélection des caractéristiques, pour laquelle, dans cette mise à jour, j'utilise l'Analyse en Composantes Principales (PCA) pour réduire le nombre de variables et conserver les informations les plus pertinentes.

L'Analyse en Composantes Principales (PCA) est une technique de réduction de dimensionnalité. Elle transforme les variables originales en un nouvel ensemble de variables non corrélées appelées composantes principales, ordonnées par la quantité de variance qu'elles capturent depuis les données. En utilisant PCA, nous pouvons éliminer efficacement les caractéristiques moins importantes, améliorant ainsi l'interprétabilité et les performances du modèle.

### Nouvelles Méthodes de Validation

Le modèle a été mis à jour pour incorporer de nouvelles méthodes de validation afin d'améliorer l'exactitude et la fiabilité, en utilisant des techniques de validation croisée et de validation croisée leave-one-out.

### 1. Validation Croisée  

La validation croisée est une méthode statistique utilisée pour estimer la performance des modèles d'apprentissage automatique. Elle implique de partitionner les données en sous-ensembles, d'entraîner le modèle sur certains sous-ensembles tout en le validant sur d'autres. Ce processus est répété plusieurs fois pour améliorer la précision de l'évaluation du modèle. La forme la plus courante est la validation croisée k-fold, où le jeu de données est divisé en 'k' sous-ensembles, et le modèle est entraîné et validé 'k' fois, chaque fois en utilisant un sous-ensemble différent comme ensemble de validation.

### 2. Validation Croisée Leave-One-Out (LOOCV)  

La validation croisée Leave-One-Out est un cas particulier de validation croisée où le nombre de sous-ensembles est égal au nombre de points de données dans le jeu de données. Cela signifie que pour chaque itération, tous les points de données sauf un sont utilisés pour l'entraînement, et le point de données laissé de côté est utilisé pour la validation. Cette méthode fournit une évaluation approfondie du modèle mais peut être coûteuse en calcul pour les grands jeux de données.

### Résultats  

Voici les résultats des différentes métriques de validation et de performance pour le modèle :  

- **Holdout :** 0.7767  
- **Échantillonnage aléatoire répété :** 0.7767  
- **Validation croisée :** 0.7794  
- **Validation croisée leave-one-out :** 0.7794  
- **Test Naive Bayes :** 0.7813    

Ces résultats indiquent que les performances du modèle se sont améliorées avec la mise en œuvre des techniques de validation mises à jour. 

### Modèle Random Forest

Le modèle Random Forest est une méthode d'apprentissage ensembliste qui utilise une collection d'arbres de décision pour améliorer la précision prédictive et contrôler le surajustement. Il fonctionne en construisant plusieurs arbres pendant l'entraînement et en sortant le mode de leurs prédictions pour les tâches de classification ou la moyenne des prédictions pour la régression.

#### Avantages du Random Forest

- **Robustesse :** Le Random Forest est moins sujet au surajustement comparé aux arbres de décision individuels, le rendant plus fiable pour divers jeux de données.  
- **Haute Exactitude :** Il fournit souvent une haute exactitude dans les prédictions, surtout dans des scénarios de données complexes et non linéaires.  
- **Importance des Caractéristiques :** Le modèle peut évaluer l'importance des différentes caractéristiques, fournissant des insights sur les variables qui impactent significativement les prédictions.  
- **Polyvalence :** Adapté à la fois pour les tâches de classification et de régression.

#### Évaluation du Modèle  

Cette implémentation inclut une évaluation robuste du modèle en utilisant la validation croisée. Voici les résultats des performances du modèle :   
- **Exactitude (Accuracy):** 0.83  
- **Rappel (Sensibilité ou TPR) :** 0.83  
- **Précision :** 0.82  
- **Score F1 :** 0.82    
- **Score AUC :** 0.86  
- **Score de Validation Croisée :** 0.74

Ces métriques indiquent que le modèle Random Forest performe bien sur le jeu de données, atteignant une exactitude de 0.83, ce qui représente une amélioration significative par rapport au modèle Naive Bayes précédent, qui avait une exactitude de seulement 0.78. De plus, le modèle Random Forest équilibre efficacement la précision et le rappel, démontrant sa robustesse en comparaison avec l'approche Naive Bayes. Cela montre la valeur ajoutée de l'utilisation de méthodes ensemblistes pour de meilleures performances prédictives.

- **Note de Mise à Jour :**  
Dans cette mise à jour, la courbe ROC pour le modèle Naive Bayes a également été actualisée pour refléter les derniers résultats et fournir une comparaison plus claire des performances des modèles.

### Optimisation du Random Forest

Cette mise à jour implémente un modèle Random Forest utilisant GridSearchCV pour l'optimisation des hyperparamètres. L'objectif est d'améliorer les performances du modèle en termes d'exactitude, sensibilité, et autres métriques d'évaluation.

#### Optimisation du Modèle avec GridSearchCV

GridSearchCV est un outil puissant fourni par la bibliothèque `scikit-learn` qui nous permet de rechercher systématiquement à travers plusieurs combinaisons d'hyperparamètres pour un modèle d'apprentissage automatique afin de trouver le meilleur ensemble performant.

- **Fonctionnement** :  
   - **Grille de Paramètres** : Les utilisateurs définissent une grille d'hyperparamètres qu'ils souhaitent explorer. Par exemple, dans un modèle Random Forest, les paramètres peuvent inclure le nombre d'arbres dans la forêt (`n_estimators`), la profondeur maximale de chaque arbre (`max_depth`), et le nombre minimum d'échantillons requis pour diviser un nœud interne (`min_samples_split`). 
   - **Validation Croisée** : Pour chaque combinaison de paramètres dans la grille, GridSearchCV effectue une validation croisée, ce qui signifie qu'il divise les données d'entraînement en plusieurs sous-ensembles (folds). Le modèle est entraîné sur certains folds et validé sur les autres. Ce processus se répète pour chaque combinaison d'hyperparamètres pour assurer une évaluation complète.
   - **Métrique de Performance** : La performance de chaque modèle est évaluée en utilisant une métrique spécifiée (par exemple, l'exactitude, le score F1) pour déterminer quelle combinaison donne les meilleures performances.

- **Avantages de GridSearchCV** :  
     - **Recherche Exhaustive** : En recherchant toutes les combinaisons des hyperparamètres spécifiés, GridSearchCV garantit que vous trouvez les paramètres optimaux pour votre modèle.  
     - **Amélioration du Modèle** : L'optimisation des hyperparamètres peut conduire à des améliorations significatives des performances du modèle, car le modèle peut devenir plus adapté à la structure sous-jacente des données.  
     - **Prévention du Surajustement** : En utilisant la validation croisée, GridSearchCV aide à évaluer comment le modèle généralise à un jeu de données indépendant, réduisant ainsi le risque de surajustement.  
     - **Intégration Facile** : GridSearchCV peut être facilement intégré dans le flux de travail d'entraînement du modèle dans `scikit-learn`, le rendant simple à implémenter. 

#### Évaluation du Modèle

Après l'entraînement et l'optimisation du modèle, ses performances sont évaluées en utilisant diverses métriques comme l'Exactitude, le Rappel, la Précision, le Score F1 et le score AUC. Résultats :
- **Exactitude (Accuracy)** : 0.83  
- **Rappel (Sensibilité / TPR)** : 0.83  
- **Précision** : 0.82  
- **Score F1** : 0.82  
- **Score AUC Après Optimisation** : 0.87

Malgré l'optimisation du modèle avec GridSearchCV, les résultats finaux pour l'exactitude, la sensibilité, la précision, le score F1 et le score AUC sont restés les mêmes que ceux du modèle initial. Cela suggère qu'il pourrait être nécessaire de revoir la sélection des caractéristiques ou d'envisager d'autres algorithmes pour des améliorations supplémentaires.

## Instructions d'Utilisation

Pour utiliser ce projet, clonez le dépôt en utilisant la commande suivante dans votre terminal ou invite de commande :

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/AIDEN243W/Prediction-du-risque-de-diabete.git
   ```
Ensuite, ouvrez le fichier Jupyter Notebook (généralement avec une extension .ipynb) en utilisant Jupyter Notebook.   

2. Pour démarrer Jupyter Notebook :
   ```bash
   jupyter notebook
   ```

## Travaux Futurs

Les améliorations et directions futures pour ce projet incluent :
- L'exploration d'autres algorithmes de classification tels que Random Forest, K-Nearest Neighbors et plus encore.
- L'optimisation des hyperparamètres pour améliorer les performances du modèle.
- L'incorporation de caractéristiques supplémentaires pour améliorer la précision des prédictions.

