import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Models.kmean import kmean


def is_numeric_string(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def main(kmeans=True):
    # Download latest version
    path = kagglehub.dataset_download("emirhanai/social-media-usage-and-emotional-well-being")
    dataframe = pd.read_csv(path + "/train.csv")
    dataframe = dataframe.drop(['Dominant_Emotion'], axis=1)
    dataframe = dataframe.dropna()
    # Fix some issues in the dataframe
    dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Age', 'Gender']] = \
        dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Gender', 'Age']].values
    if kmeans:
        distance = []
        # K mean testing
        for j in range(1, 11):
            Kmean = kmean(dataframe=dataframe, k=j)
            for i in range(10000):
                print(f'Epoch: {i}')
                Kmean.find_closest_centroid()
                if Kmean.updateCentroids():
                    break
            distance.append(Kmean.averageDistanceToCentroid())

        plt.plot(distance, 'o-r')
        plt.xticks(np.arange(len(distance)), np.arange(1, len(distance)+1))
        plt.title('K vs Average Distance')
        plt.xlabel('K')
        plt.ylabel('Average Distance')
        plt.show()


if __name__ == '__main__':
    main()
