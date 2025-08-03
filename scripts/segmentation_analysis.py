import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('outputs/cleaned_data.csv')

# Features selected for segmentation
selected_features = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['pca1'], df['pca2'] = components[:, 0], components[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2')
plt.title('Customer Segments (PCA)')
plt.savefig('outputs/charts/cluster_visualization.png')
plt.close()

# Save clustered data
df.to_csv('outputs/cleaned_data.csv', index=False)
print("Clustering complete and saved.")
