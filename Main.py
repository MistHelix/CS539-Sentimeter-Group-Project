import kagglehub
import pandas as pd


def main():
    # Download latest version
    path = kagglehub.dataset_download("emirhanai/social-media-usage-and-emotional-well-being")

    dataframe = pd.read_csv(path+"/train.csv")
    dataframe = dataframe.drop(['Dominant_Emotion'], axis=1)
    print(dataframe)


if __name__ == '__main__':
    main()