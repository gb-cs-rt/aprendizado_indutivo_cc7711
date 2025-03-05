import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.io import arff

data, meta = arff.loadarff('./grub-damage.arff')

# Convert byte strings to regular strings
data = pd.DataFrame(data)
for column in data.select_dtypes([np.object_]):
    data[column] = data[column].str.decode('utf-8')

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical features
year_zone = le.fit_transform(data['year_zone']).reshape(-1, 1)
year = np.asarray(data['year']).reshape(-1, 1)
strip = np.asarray(data['strip']).reshape(-1, 1)
pdk = np.asarray(data['pdk']).reshape(-1, 1)
damage_rankRJT = le.fit_transform(data['damage_rankRJT']).reshape(-1, 1)
damage_rankALL = le.fit_transform(data['damage_rankALL']).reshape(-1, 1)
dry_or_irr = le.fit_transform(data['dry_or_irr']).reshape(-1, 1)
zone = le.fit_transform(data['zone']).reshape(-1, 1)

features = np.concatenate((year_zone, year, strip, pdk, damage_rankRJT, damage_rankALL, dry_or_irr, zone), axis=1)

# Encode target with specified order
target_order = ['low', 'average', 'high', 'veryhigh']
data['GG_new'] = pd.Categorical(data['GG_new'], categories=target_order, ordered=True)
target = data['GG_new'].cat.codes

# Check class distribution before and after encoding
print("Class distribution before encoding:")
print(data['GG_new'].value_counts())
print("Class distribution after encoding:")
print(pd.Series(target).value_counts())

Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(20, 13))
tree.plot_tree(Arvore, feature_names=['year_zone', 'year', 'strip', 'pdk', 'damage_rankRJT', 'damage_rankALL', 'dry_or_irr', 'zone'], class_names=target_order, filled=True, rounded=True)
plt.savefig('tree.png', format='png', dpi=550)

fix, ax = plt.subplots(figsize=(15, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore, features, target, display_labels=target_order, values_format='d', ax=ax)
plt.savefig('confusion_matrix.png', format='png', dpi=300)
