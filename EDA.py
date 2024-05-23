class EDA:
    def __init__(self, data):
        self.data = data

    def clean_dataset(self):
        # print(self.data)
        # print(self.data.shape)
        # print(self.data.head())
        # print(self.data.tail())
        # print(type(self.data.info()))
        # print(self.data.isnull().sum())
        # print((self.data.isnull().sum()/(len(self.data)))*100)
        # print(self.data.describe())

        # Remove null columns from data
        self.data.dropna(how='all', axis=1, inplace=True)

        # find non-numeric column names
        obj_cols = list(self.data.select_dtypes(include=['object']).columns)
        # print(obj_cols)

        # check if they are useful
        for column in obj_cols:
            unique_num = len(self.data[column].unique())
            total_num = len(self.data[column])
            uniqueness = (unique_num / total_num) * 100
            # if most of the data is unique, then it's not useful so remove it
            if uniqueness > 90:
                print(column)
                self.data.drop(column, axis=1, inplace=True)
            # otherwise it's useful, so convert to numeric
            else:
                # map each unique value to it's index
                unique_vals = self.data[column].unique()
                unique_indexes = range(0, len(unique_vals))
                self.data[column] = self.data[column].map(dict(zip(unique_vals, unique_indexes)))

        return

    def scale_features(self):
        # # Min-Max Normalization
        # self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        # or

        # Standardize features by removing the mean and scaling to unit variance
        # x = (x - m) / v
        variance = self.data.var()
        mean = self.data.mean()
        self.data = (self.data - mean) / variance

        return

    def get_data(self):
        return self.data
