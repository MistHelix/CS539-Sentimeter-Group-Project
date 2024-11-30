import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Models.gmm import GMMCluster


def is_numeric_string(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def main():
    dataframe = pd.read_csv("train.csv")
    dataframe = dataframe.drop(['Dominant_Emotion'], axis=1)
    dataframe = dataframe.dropna()

    # Fix some issues in the dataframe
    dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Age', 'Gender']] = \
        dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Gender', 'Age']].values

    # Factorize categorical columns
    dataframe['Gender'], _ = pd.factorize(dataframe['Gender'])
    dataframe['Platform'], _ = pd.factorize(dataframe['Platform'])
    dataframe = dataframe.drop('User_ID', axis=1).astype(float)

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