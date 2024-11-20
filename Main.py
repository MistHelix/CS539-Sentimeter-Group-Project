import kagglehub
import pandas as pd

from Models.kmean import kmean


def is_numeric_string(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def main():
    # Download latest version
    path = kagglehub.dataset_download("emirhanai/social-media-usage-and-emotional-well-being")

    dataframe = pd.read_csv(path+"/train.csv")
    dataframe = dataframe.drop(['Dominant_Emotion'], axis=1)
    print(dataframe)
    dataframe = dataframe.dropna()
    # Fix some issues in the dataframe
    dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Age', 'Gender']] = \
    dataframe.loc[dataframe['Gender'].apply(is_numeric_string), ['Gender', 'Age']].values

    # K mean testing
    for j in range(1,20):
        Kmean = kmean(dataframe=dataframe, k=j)
        for i in range(10000):
            print(f'Epoch: {i}')
            Kmean.find_closest_centroid()
            if Kmean.updateCentroids():
                break
        Kmean.averageDistanceToCentroid()



if __name__ == '__main__':
    main()