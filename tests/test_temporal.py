from src.analytics.temporal import TemporalConfirm


def test_not_confirmed_until_window_is_full():
    tc = TemporalConfirm(3, 5)
    assert [tc.update(True) for _ in range(4)] == [False] * 4
    assert tc.update(True)


def test_below_n_is_not_confirmed():
    tc = TemporalConfirm(3, 5)
    results = [tc.update(d) for d in [True, False, True, False, False]]
    assert not results[-1]  # 2 из 5


def test_exactly_n_is_confirmed():
    tc = TemporalConfirm(3, 5)
    results = [tc.update(d) for d in [True, False, True, False, True]]
    assert results[-1]  # 3 из 5


def test_old_frames_drop_out_of_window():
    tc = TemporalConfirm(3, 5)
    for _ in range(5):
        tc.update(True)
    for _ in range(5):
        last = tc.update(False)
    assert not last