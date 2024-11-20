import math

import pandas as pd


class kmean:

    def __init__(self, dataframe, k):
        self.k = k
        dataframe['Gender'], unique = pd.factorize(dataframe['Gender'])
        dataframe['Platform'], unique = pd.factorize(dataframe['Platform'])
        dataframe = dataframe.drop('User_ID', axis=1)
        dataframe['Centroid'] = 0
        dataframe = dataframe.astype(float)
        self.dataframe = dataframe
        self.centroids = dataframe.sample(k, ignore_index=True)
        self.centroids.drop(['Centroid'], axis=1)
        self.lastcentroids = self.centroids.copy()

    def find_closest_centroid(self):
        for row in self.dataframe.itertuples():
            bestcentroid = -1
            disttoclosestcentroid = float('inf')
            for centroid in range(self.k):
                sum = 0
                for i in range(1,len(row)-2):
                    sum += math.pow(self.centroids.iloc[centroid].iloc[i-1]-row[i], 2)
                distance = math.sqrt(sum)
                if disttoclosestcentroid > distance:
                    bestcentroid = centroid
                    disttoclosestcentroid = distance
            self.dataframe.loc[row[0], 'Centroid'] = bestcentroid

    def updateCentroids(self):
        for centroid in range(self.k):
            closest = self.dataframe[self.dataframe['Centroid'] == centroid]
            self.centroids.loc[centroid] = closest.iloc[:, :-1].mean()
        if self.lastcentroids.equals(self.centroids):
            return True
        else:
            self.lastcentroids = self.centroids.copy()
            return False

    def averageDistanceToCentroid(self):
        average_distance = 0
        for row in self.dataframe.itertuples():
            centroid = int(row[9])
            sum = 0
            for i in range(1,len(row)-2):
                sum += math.pow(self.centroids.iloc[centroid].iloc[i-1]-row[i], 2)
            average_distance += math.sqrt(sum)
        print("K is ", self.k, " Distance:",average_distance/len(self.dataframe))