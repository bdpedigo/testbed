# %%
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

features = pd.read_csv("data/claire/features_for_ben.csv", index_col=0).values
labels = pd.read_csv("data/claire/labels_for_ben.csv", index_col=0).values.ravel()
# %%

import time

currtime = time.time()
lda = LinearDiscriminantAnalysis()
lda.fit(features, labels)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

from sklearn.metrics import accuracy_score

preds = lda.predict(features)
print(f"Accuracy: {accuracy_score(labels, preds):.3f}")
# %%
