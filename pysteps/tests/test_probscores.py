# -*- coding: utf-8 -*-
import pytest
import numpy as np
from pysteps.verification.probscores import CRPS_init
from pysteps.verification.probscores import reldiag_init
from pysteps.verification.probscores import ROC_curve_init

from pysteps.verification.probscores import CRPS_compute
from pysteps.verification.probscores import reldiag_compute


from numpy.testing import assert_array_almost_equal


def test_CRPS_init():
    assert CRPS_init() == {"CRPS_sum": 0.0, "n": 0.0}


def test_CRPS_compute():
    a_crps = {"CRPS_sum": 1000., "n": 100.}
    assert_array_almost_equal(CRPS_compute(a_crps), 10.0)


test_reldiag_init_data = [
    ("X_min", 2.0),
    ("bin_edges", np.linspace(-1e-6, 1+1e-6, 11)),
    ("n_bins", 10),
    ("X_sum", np.zeros(10)),
    ("Y_sum", np.zeros(10)),
    ("num_idx", np.zeros(10)),
    ("sample_size", np.zeros(10)),
    ("min_count", 10),
    ]


@pytest.mark.parametrize("variable, expected", test_reldiag_init_data)
def test_reldiag_init(variable, expected):
    r_ini = reldiag_init(2.0)
    assert_array_almost_equal(r_ini[variable], expected)


def test_reldiag_compute_r():
    a_reldiag = {"Y_sum": 1000.,
                 "num_idx": 100.,
                 "X_sum": 2000.,
                 }
    r, _ = reldiag_compute(a_reldiag)
    assert_array_almost_equal(r, 20.0)


def test_reldiag_compute_f():
    a_reldiag = {"Y_sum": 1000.,
                 "num_idx": 100.,
                 "X_sum": 2000.,
                 }
    _, f = reldiag_compute(a_reldiag)
    assert_array_almost_equal(f, 10.0)


test_roc_curve_init_data = [
    ("X_min", 2.0),
    ("hits", np.zeros(10)),
    ("misses", np.zeros(10)),
    ("false_alarms", np.zeros(10)),
    ("corr_neg", np.zeros(10)),
    ("prob_thrs", np.linspace(0.0, 1.0, 10)),
    ]


@pytest.mark.parametrize("variable, expected", test_roc_curve_init_data)
def test_roc_curve_init(variable, expected):
    roc_ini = ROC_curve_init(2.0)
    assert_array_almost_equal(roc_ini[variable], expected)


# # CREATE DATASETS TO TEST

# a = np.arange(9, dtype=float).reshape(3, 3)
# b = np.tile(a, (4, 1, 1))
# b1 = b.copy()
# b1[3] = np.nan
# a1 = a.copy()
# a1[:] = np.nan
# a2 = a.copy()
# a2[0, :] = np.nan

# #  test data
# test_data = [
#     (a, False, None, a),
#     (b, False, None, a),
#     (b1, True, None, a),
#     (b1, False, None, a1),
#     (b, False, 0.0, a),
#     (b, False, 3.0, a2),
#     (b, True, 3.0, a2),
#     (b1, True, 3.0, a2),
#     ]


# @pytest.mark.parametrize("X, ignore_nan, X_thr, expected", test_data)
# def test_ensemblestats_mean(X, ignore_nan, X_thr, expected):
#     """Test ensemblestats mean."""
#     assert_array_almost_equal(mean(X, ignore_nan, X_thr), expected)


# #  test exceptions
# test_exceptions = [(0), (None), (a[0, :]),
#                    (np.tile(a, (4, 1, 1, 1))),
#                    ]


# @pytest.mark.parametrize("X", test_exceptions)
# def test_exceptions_mean(X):
#     with pytest.raises(Exception):
#         mean(X)


# #  test data
# b2 = b.copy()
# b2[2, 2, 2] = np.nan

# test_data = [
#     (b, 2.0, False, np.array([[0., 0., 1.], [1., 1., 1.], [1., 1., 1.]])),
#     (b2, 2.0, False, np.array([[0., 0., 1.], [1., 1., 1.], [1., 1., np.nan]])),
#     (b2, 2.0, True, np.array([[0., 0., 1.], [1., 1., 1.], [1., 1., 1.]])),
#     ]


# @pytest.mark.parametrize("X, X_thr, ignore_nan, expected", test_data)
# def test_ensemblestats_excprob(X, X_thr, ignore_nan, expected):
#     """Test ensemblestats excprob."""
#     assert_array_almost_equal(excprob(X, X_thr, ignore_nan), expected)


# #  test exceptions
# test_exceptions = [(0), (None),
#                    (a[0, :]),
#                    (a),
#                    ]


# @pytest.mark.parametrize("X", test_exceptions)
# def test_exceptions_excprob(X):
#     with pytest.raises(Exception):
#         excprob(X, 2.0)
