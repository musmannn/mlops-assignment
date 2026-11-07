from calc import add, mean


def test_add():
    assert add(2, 3) == 5


def test_mean():
    # RED run: change the expected value to 3 and push -> both matrix jobs fail.
    # GREEN run: change it back to 2 and push.
    assert mean([1, 2, 3]) == 2
