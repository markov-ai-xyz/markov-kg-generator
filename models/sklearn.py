from sklearn.cluster import AgglomerativeClustering
import numpy as np


def cluster_spans(spans, embedding_generator, n_clusters=10):
    embeddings = np.array([embedding_generator(span.text) for span in spans])
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)
    return labels
