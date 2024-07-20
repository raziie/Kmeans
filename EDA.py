import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class EDA:
    def __init__(self, data):
        self.data = data

    def clean_dataset(self):
        # Remove null columns from data
        self.data.dropna(how='all', axis=1, inplace=True)

        # find non-numeric column names
        obj_cols = list(self.data.select_dtypes(include=['object']).columns)

        # check if they are useful
        for column in obj_cols:
            unique_num = len(self.data[column].unique())
            total_num = len(self.data[column])
            uniqueness = (unique_num / total_num) * 100
            # if most of the data is unique, then it's not useful so remove it
            if uniqueness > 90:
                self.data.drop(column, axis=1, inplace=True)
            # otherwise it's useful, so convert to numeric
            else:
                # map each unique value to it's index
                unique_vals = self.data[column].unique()
                unique_indexes = range(0, len(unique_vals))
                self.data[column] = self.data[column].map(dict(zip(unique_vals, unique_indexes)))

                # or

                # label_encoder = LabelEncoder()
                # self.data[column] = label_encoder.fit_transform(self.data[column])

        return

    def remove_outliers(self):
        for column in self.data.columns:
            # calculate IQR for each column
            Q1 = self.data[column].quantile(0.05)
            Q3 = self.data[column].quantile(0.95)
            IQR = Q3 - Q1

            # identify outliers
            threshold = 1.5
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            outliers = self.data[(self.data[column] < lower) | (self.data[column] > upper)]

            # remove outliers
            self.data.drop(outliers.index, inplace=True)

    def scale_features(self):
        # Min-Max Normalization
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        # or

        # # Standardize features by removing the mean and scaling to unit variance
        # # x = (x - m) / v
        # variance = self.data.var()
        # mean = self.data.mean()
        # self.data = (self.data - mean) / variance

        # or

        # column_list = self.data.columns.tolist()
        # features = self.data[column_list]
        # scaler = MinMaxScaler()
        # scaled_features = scaler.fit_transform(features)
        # self.data = pd.DataFrame(scaled_features, columns=self.data.columns)

        return

    def get_data(self):
        return self.data
