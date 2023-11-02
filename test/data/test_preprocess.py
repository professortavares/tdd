import pandas as pd
import numpy as np
import pytest

from src.data.preprocess import Preprocess as pp
from src.exceptions.preprocess import PreprocessException

def test_normalize_collumns_verify_collumns_sucess():
    """
    Test scenario: check whether the columns present in the dataframe are
    consistent with expectations

    Assumptions: dataframe columns are correct

    Expected result: the method executes correctly and no exceptions are raised
    """

    # Create dataframe with correct columns:
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    # and random values greater than 0

    #########
    # setup #
    #########
    # Prepares the test environment (preconditions)
    df = pd.DataFrame(np.random.randint(1, 100, size=(100, 9)), columns=[
                        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

    #############
    # execution #
    #############
    # Executes the test
    result = pp.normalize_collumns(df)

    ###########
    # asserts #
    ###########
    # Checks that all post conditions have been satisfied

    # the dataframe exists
    assert result is not None
    # the dataframe is not empty
    assert not result.empty
    # the dataframe has the same number of columns as the original dataframe
    assert len(result.columns) == len(df.columns)
    # the dataframe has the same columns as the original dataframe
    assert all(result.columns == df.columns)
    # all the values in the dataframe are greater than 0 and less than 1
    assert all(result.values >= 0) and all(result.values <= 1)



def test_normalize_collumns_more_columns():
    """
    Test scenario: detects that the sent dataframe has more columns than expected

    Assumptions: the informed dataframe has **more columns**

    Expected result: the method should raise an exception
    """

    # Create dataframe with 1 more column than expected:
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome, OtherCollumn
    # and random values greater than 0

    #########
    # setup #
    #########
    # Prepares the test environment (preconditions)
    df = pd.DataFrame(np.random.randint(1, 100, size=(100, 10)), columns=[
                        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'OtherCollumn'])

    #############
    # execution #
    #############
    # Executes the test
    with pytest.raises(PreprocessException) as e:
        _ = pp.normalize_collumns(df)
    assert str(e.value) == "more columns than expected"


def test_normalize_collumns_less_columns():
    """
    Test scenario: detects that the sent dataframe has less columns than expected

    Assumptions: the informed dataframe has **les columns**

    Expected result: the method should raise an exception
    """

    # Create dataframe with 1 less column than expected:
    # Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome, OtherCollumn
    # and random values greater than 0

    #########
    # setup #
    #########
    # Prepares the test environment (preconditions)
    df = pd.DataFrame(np.random.randint(1, 100, size=(100, 10)), columns=[
                        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'OtherCollumn'])

    #############
    # execution #
    #############
    # Executes the test
    with pytest.raises(PreprocessException) as e:
        _ = pp.normalize_collumns(df)
    assert str(e.value) == "less columns than expected"

def test_normalize_collumns_wrong_values():
    """
    Test scenario: Verify that the values for preprocessing are consistent

    Assumptions: the reported dataframe has an inconsistency in the pregnancy value.
    Some of the values **are negative**

    Expected result: the method should raise an exception
    """

    # Create dataframe with correct columns:
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    # and random values greater than 0, but pregnancies with some negative values

    #########
    # setup #
    #########
    # Prepares the test environment (preconditions)
    df = pd.DataFrame(np.random.randint(1, 100, size=(100, 9)), columns=[
                        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    # some pregnancies with negative values, to simulate an error
    # if the value of pregancies is less than 5, then it will be negative
    df.loc[df['Pregnancies'] < 5, 'Pregnancies'] = -1

    #############
    # execution #
    #############
    # Executes the test
    with pytest.raises(PreprocessException) as e:
        _ = pp.normalize_collumns(df)
    assert str(e.value) == "wrong values in the dataframe"



