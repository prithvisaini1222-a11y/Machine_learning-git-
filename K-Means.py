import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [
    [20, 30], [25, 35], [30, 40],
    [70, 80], [75, 85], [80, 90]
]

wcss = []

for k in range(1, 6):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 6), wcss, marker='o')

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
