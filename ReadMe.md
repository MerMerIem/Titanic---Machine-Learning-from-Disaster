# Titanic - Machine Learning from Disaster
## Problem Definition
**Objective:** This objective of this project is to build a model that can predict wether a passenger can survive or not based on some features.

**Target variable:** A column named **Survived**, it is represented in binary 1 if the passenger survived and 0 instead.

**Evaluation metric:** **F1-score** Where beta=1 perfectly equal weight — chosen because: both FP and FN are costy in this case, FP costs false hope and emotional damage, while the FN costs premature decisions (inheritance, etc.), practical/legal cost.

**Success criteria:** 
- Beat trivial baseline of F1 = 0 (majority-class prediction) — sanity check.
- Beat simple-model baseline of F1 = [score from plain Logistic Regression] — the real bar.
- Target score: [Y] — set once you see how much headroom there is between the two.

**Constraints:** This project is purely kaggle exercise where only the last file (test.csv) is asked to submit along with the predictions (this will show how good is the model).

**Data notes:** 
- This data is from a competition in kaggle https://www.kaggle.com/competitions/titanic/overview
- Data is taken from the disaster in 1912 where the boat named titanic drowned due to an iceberg so many passengers passed away, so later on if any similar case happens the model that will be built using these data would help give good results however not in any other disaster. 

## Data Understanding

- This data was taken from historical disaster so the study will be useful only on a similar case not any universal disaster.
- **Hypothesis to verify:* There is a bias on the column Cabin, for now we assume that the passengers with class = 3 has no Cabin.