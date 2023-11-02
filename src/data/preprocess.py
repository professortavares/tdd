import pandas as pd
from src.exceptions.preprocess import PreprocessException

class Preprocess:
    # Class responsible for preprocessing the data

    @staticmethod
    def normalize_collumns(df: pd.DataFrame):
        """
        Normalize the collumns of the dataframe

        Params:
        -------
            - df: pd.DataFrame
                Dataframe to be normalized
        Return:
        -------
            - pd.DataFrame
                Dataframe normalized

        Exemple of use:
        ---------------
            >>> import pandas as pd
            >>> from src.data.preprocess import Preprocess as pp
            >>> df = pd.DataFrame(np.random.randint(1, 100, size=(100, 9)), columns=[
            ...                     'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
            >>> result = pp.normalize_collumns(df)
            >>> result
        """

        # Validation of the dataframe
        # - check if the dataframe is not empty
        if df.empty:
            raise PreprocessException("The dataframe is empty")

        # - check if the dataframe has the expected columns
        expected_columns = {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'}
        current_columns = set(df.columns)
        r = expected_columns - current_columns
        if len(r) > 0:
            raise PreprocessException("less columns than expected")

        r = current_columns - expected_columns
        if len(r) > 0:
            raise PreprocessException("more columns than expected")

        # normalize the dataframe
        df = (df - df.min()) / (df.max() - df.min())

        return df