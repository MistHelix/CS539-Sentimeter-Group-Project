import numpy as np
import pandas as pd
from numpy.linalg._umath_linalg import svd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Models.gmm import GMMCluster
from Models.svd import SVDModel
import kagglehub

def is_numeric_string(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def main():
    path = kagglehub.dataset_download("emirhanai/social-media-usage-and-emotional-well-being")
    dataframe = pd.read_csv(path + "/train.csv")
    dataframe = dataframe.drop(['Dominant_Emotion'], axis=1)
    dataframe = dataframe.dropna()

    # Fix some issues in the dataframe
    dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Age', 'Gender']] = \
        dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Gender', 'Age']].values

    svd_model = SVDModel(dataframe=dataframe, n_components=5)  # Adjust n_components as needed
    svd_model.fit()
    reduced_data = svd_model.transform()

    # Factorize categorical columns
    dataframe['Gender'], _ = pd.factorize(dataframe['Gender'])
    dataframe['Platform'], _ = pd.factorize(dataframe['Platform'])
    dataframe = dataframe.drop('User_ID', axis=1).astype(float)

    # Initialize SVD
    svd_model = SVDModel(dataframe=dataframe, n_components=5)  # Adjust n_components as needed
    svd_model.fit()
    reduced_data = svd_model.transform()

    # Iterate over different values of k
    results = []
    for j in range(2, 300):  # Testing for a smaller range of k for quick results
        print(f"\nTesting GMM with k = {j}")
        gmm = GMMCluster(dataframe=dataframe, k=j)
        gmm.fit()
        predictions = gmm.predict()

        # Evaluate clustering performance
        silhouette = silhouette_score(dataframe, predictions)
        calinski_harabasz = calinski_harabasz_score(dataframe, predictions)
        davies_bouldin = davies_bouldin_score(dataframe, predictions)

        print(f"Silhouette Score for k = {j}: {silhouette:.3f}")
        print(f"Calinski-Harabasz Index for k = {j}: {calinski_harabasz:.3f}")
        print(f"Davies-Bouldin Index for k = {j}: {davies_bouldin:.3f}")

        # Append results for later analysis
        results.append({
            'k': j,
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        })

    # Display results for all k values
    results_df = pd.DataFrame(results)
    print("\nPerformance metrics for different k values:")
    print(results_df)
    # Save results for further analysis if needed
    results_df.to_csv("clustering_metrics.csv", index=False)

if __name__ == '__main__':
    main()