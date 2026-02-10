import numpy as np

from niarb import parsing


def test_matrix():
    # This test case is taken from the github action matrix strategy example
    # https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
    matrix = {
        "fruit": ["apple", "pear"],
        "animal": ["cat", "dog"],
    }
    include = [
        {"color": "green"},
        {"color": "pink", "animal": "cat"},
        {"fruit": "apple", "shape": "circle"},
        {"fruit": "banana"},
        {"fruit": "banana", "animal": "cat"},
    ]
    expected = [
        {"fruit": "apple", "animal": "cat", "color": "pink", "shape": "circle"},
        {"fruit": "apple", "animal": "dog", "color": "green", "shape": "circle"},
        {"fruit": "pear", "animal": "cat", "color": "pink"},
        {"fruit": "pear", "animal": "dog", "color": "green"},
        {"fruit": "banana"},
        {"fruit": "banana", "animal": "cat"},
    ]
    out = parsing.matrix(matrix, include)
    assert out == expected


def test_array():
    out = parsing.array("2,1:5,3")
    expected = np.array([2, 1, 2, 3, 4, 3])
    np.testing.assert_equal(out, expected)


def test_indices():
    out = parsing.indices("")
    expected = []
    assert out == expected

    out = parsing.indices("1,3,5-7")
    expected = [1, 3, 5, 6, 7]
    assert out == expected
