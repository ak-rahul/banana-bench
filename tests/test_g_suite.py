"""
Tests for bananabench.g_suite: the complete constrained G-suite (g01-g24).

Each function returns (objective, inequality_violations, equality_violations)
rather than a single float; correctness is checked against the documented
optimal objective values and feasibility of the documented optimal points,
using the exact vectors published in the CEC 2006 technical report
(docs-bench/). g23's published optimal vector is truncated/corrupted in the
source PDF's text extraction (its 4th equality constraint doesn't check out
against it); the point used below was reconstructed from the other three
equality constraints and independently verified to land within 0.002 of the
documented optimum. g20 is a known non-strict case: the report itself notes
its "best known" point is slightly infeasible, so its test only checks the
objective value and that violations stay within the CEC epsilon (1e-4)
rather than full feasibility.
"""

import numpy as np
import pytest

from bananabench import g_suite


class TestG01:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g01(np.ones(12))

    def test_return_shape(self):
        f, g, h = g_suite.g01(np.ones(13))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (9,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1], dtype=float)
        f, g, h = g_suite.g01(x)
        assert f == pytest.approx(-15.0, abs=1e-6)
        assert np.all(g <= 1e-6)


class TestG02:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g02(np.ones(19))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                3.16246061572185,
                3.12833142812967,
                3.09479212988791,
                3.06145059523469,
                3.02792915885555,
                2.99382606701730,
                2.95866871765285,
                2.92184227312450,
                0.49482511456933,
                0.48835711005490,
                0.48231642711865,
                0.47664475092742,
                0.47129550835493,
                0.46623099264167,
                0.46142004984199,
                0.45683664767217,
                0.45245876903267,
                0.44826762241853,
                0.44424700958760,
                0.44038285956317,
            ]
        )
        f, g, h = g_suite.g02(x)
        assert f == pytest.approx(-0.80361910412559, abs=1e-6)
        assert np.all(g <= 1e-2)


class TestG03:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g03(np.ones(9))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.full(10, 0.31624357647283069)
        f, g, h = g_suite.g03(x)
        assert f == pytest.approx(-1.00050010001000, abs=1e-2)
        assert np.all(np.abs(h) <= 1e-2)


class TestG05:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g05(np.ones(3))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [679.945148297028709, 1026.06697600004691, 0.118876369094410433, -0.39623348521517826]
        )
        f, g, h = g_suite.g05(x)
        assert f == pytest.approx(5126.4967140071, abs=1e-2)
        assert np.all(g <= 1e-2)
        assert np.all(np.abs(h) <= 1e-2)


class TestG04:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g04(np.ones(4))

    def test_return_shape(self):
        f, g, h = g_suite.g04(np.ones(5))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (6,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([78.0, 33.0, 29.995, 45.0, 36.776])
        f, g, h = g_suite.g04(x)
        assert f == pytest.approx(-30665.539, abs=1.0)
        assert np.all(g <= 1e-3)


class TestG06:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g06(np.ones(3))

    def test_return_shape(self):
        f, g, h = g_suite.g06(np.ones(2))
        assert isinstance(f, float)
        assert g.shape == (2,)
        assert h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([14.095, 0.84296])
        f, g, h = g_suite.g06(x)
        assert f == pytest.approx(-6961.81388, abs=1.0)
        assert np.all(g <= 1e-2)


class TestG07:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g07(np.ones(9))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                2.17199634142692,
                2.3636830416034,
                8.77392573913157,
                5.09598443745173,
                0.990654756560493,
                1.43057392853463,
                1.32164415364306,
                9.82872576524495,
                8.2800915887356,
                8.3759266477347,
            ]
        )
        f, g, h = g_suite.g07(x)
        assert f == pytest.approx(24.30620906818, abs=1e-2)
        assert np.all(g <= 1e-2)


class TestG09:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g09(np.ones(6))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                2.33049935147405174,
                1.95137236847114592,
                -0.477541399510615805,
                4.36572624923625874,
                -0.624486959100388983,
                1.03813099410962173,
                1.5942266780671519,
            ]
        )
        f, g, h = g_suite.g09(x)
        assert f == pytest.approx(680.630057374402, abs=1e-2)
        assert np.all(g <= 1e-2)


class TestG10:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g10(np.ones(7))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                579.306685017979589,
                1359.97067807935605,
                5109.97065743133317,
                182.01769963061534,
                295.601173702746792,
                217.982300369384632,
                286.41652592786852,
                395.601173702746735,
            ]
        )
        f, g, h = g_suite.g10(x)
        assert f == pytest.approx(7049.24802052867, abs=1e-2)
        assert np.all(g <= 1e-2)


class TestG08:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g08(np.ones(3))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([1.2279713, 4.2453733])
        f, g, h = g_suite.g08(x)
        assert f == pytest.approx(-0.095825, abs=1e-3)
        assert np.all(g <= 1e-6)


class TestG11:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g11(np.ones(3))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([-0.70711, 0.5])
        f, g, h = g_suite.g11(x)
        assert f == pytest.approx(0.7499, abs=1e-3)
        assert np.all(np.abs(h) <= 1e-4)


class TestG12:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g12(np.ones(2))

    def test_return_shape(self):
        f, g, h = g_suite.g12(np.ones(3))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (1,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([5.0, 5.0, 5.0])
        f, g, h = g_suite.g12(x)
        assert f == pytest.approx(-1.0, abs=1e-6)
        assert np.all(g <= 1e-6)

    def test_point_between_grid_spheres_is_infeasible(self):
        # (5.5, 5.5, 5.5) is equidistant from the 8 nearest grid centers and
        # farther than the 0.25 sphere radius from all of them.
        _, g, _ = g_suite.g12(np.array([5.5, 5.5, 5.5]))
        assert g[0] > 0


class TestG13:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g13(np.ones(4))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                -1.71714224003,
                1.59572124049468,
                1.8272502406271,
                -0.763659881912867,
                -0.76365986736498,
            ]
        )
        f, g, h = g_suite.g13(x)
        assert f == pytest.approx(0.053941514041898, abs=1e-4)
        assert np.all(np.abs(h) <= 1e-2)


class TestG14:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g14(np.ones(9))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                0.0406684113216282,
                0.147721240492452,
                0.783205732104114,
                0.00141433931889084,
                0.485293636780388,
                0.000693183051556082,
                0.0274052040687766,
                0.0179509660214818,
                0.0373268186859717,
                0.0968844604336845,
            ]
        )
        f, g, h = g_suite.g14(x)
        assert f == pytest.approx(-47.7648884594915, abs=1e-3)
        assert np.all(np.abs(h) <= 1e-2)


class TestG15:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g15(np.ones(2))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([3.51212812611795133, 0.216987510429556135, 3.55217854929179921])
        f, g, h = g_suite.g15(x)
        assert f == pytest.approx(961.715022289961, abs=1e-3)
        assert np.all(np.abs(h) <= 1e-2)


class TestG16:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g16(np.ones(4))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                705.174537070090537,
                68.5999999999999943,
                102.899999999999991,
                282.324931593660324,
                37.5841164258054832,
            ]
        )
        f, g, h = g_suite.g16(x)
        assert f == pytest.approx(-1.90515525853479, abs=1e-3)
        assert np.all(g <= 1e-2)


class TestG17:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g17(np.ones(5))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                201.784467214523659,
                99.9999999999999005,
                383.071034852773266,
                420.0,
                -10.9076584514292652,
                0.0731482312084287128,
            ]
        )
        f, g, h = g_suite.g17(x)
        assert f == pytest.approx(8853.53967480648, abs=1.0)
        assert np.all(np.abs(h) <= 1e-2)


class TestG18:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g18(np.ones(8))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                -0.657776192427943163,
                -0.153418773482438542,
                0.323413871675240938,
                -0.946257611651304398,
                -0.657776194376798906,
                -0.753213434632691414,
                0.323413874123576972,
                -0.346462947962331735,
                0.59979466285217542,
            ]
        )
        f, g, h = g_suite.g18(x)
        assert f == pytest.approx(-0.866025403784439, abs=1e-4)
        assert np.all(g <= 1e-2)


class TestG19:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g19(np.ones(14))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                1.66991341326291344e-17,
                3.95378229282456509e-16,
                3.94599045143233784,
                1.06036597479721211e-16,
                3.2831773458454161,
                9.99999999999999822,
                1.12829414671605333e-17,
                1.2026194599794709e-17,
                2.50706276000769697e-15,
                2.24624122987970677e-15,
                0.370764847417013987,
                0.278456024942955571,
                0.523838487672241171,
                0.388620152510322781,
                0.298156764974678579,
            ]
        )
        f, g, h = g_suite.g19(x)
        assert f == pytest.approx(32.6555929502463, abs=1e-3)
        assert np.all(g <= 1e-2)


class TestG20:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g20(np.ones(23))

    def test_documented_point_matches_objective_and_reported_epsilon_violations(self):
        # This problem's own source technical report notes that the "best
        # known" point below is itself slightly infeasible and that no
        # feasible solution has ever been verified for g20 -- so this checks
        # the objective value and that constraint violations stay within the
        # CEC epsilon (1e-4) rather than asserting strict feasibility.
        x = np.array(
            [
                1.28582343498528086e-18,
                4.83460302526130664e-34,
                0.0,
                0.0,
                6.30459929660781851e-18,
                7.57192526201145068e-34,
                5.03350698372840437e-34,
                9.28268079616618064e-34,
                0.0,
                1.76723384525547359e-17,
                3.55686101822965701e-34,
                2.99413850083471346e-34,
                0.158143376337580827,
                2.29601774161699833e-19,
                1.06106938611042947e-18,
                1.31968344319506391e-18,
                0.530902525044209539,
                0.0,
                2.89148310257773535e-18,
                3.34892126180666159e-18,
                0.0,
                0.310999974151577319,
                5.41244666317833561e-05,
                4.84993165246959553e-16,
            ]
        )
        f, g, h = g_suite.g20(x)
        assert f == pytest.approx(0.204979400, abs=1e-6)
        assert np.all(np.abs(h) <= 1e-4)


class TestG21:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g21(np.ones(6))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                193.724510070034967,
                5.56944131553368433e-27,
                17.3191887294084914,
                100.047897801386839,
                6.68445185362377892,
                5.99168428444264833,
                6.21451648886070451,
            ]
        )
        f, g, h = g_suite.g21(x)
        assert f == pytest.approx(193.724510070035, abs=1e-3)
        assert np.all(g <= 1e-2)
        assert np.all(np.abs(h) <= 1e-2)


class TestG22:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g22(np.ones(21))

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array(
            [
                236.430975504001054,
                135.82847151732463,
                204.818152544824585,
                6446.54654059436416,
                3007540.83940215595,
                4074188.65771341929,
                32918270.5028952882,
                130.075408394314167,
                170.817294970528621,
                299.924591605478554,
                399.258113423595205,
                330.817294971142758,
                184.51831230897065,
                248.64670239647424,
                127.658546694545862,
                269.182627528746707,
                160.000016724090955,
                5.29788288102680571,
                5.13529735903945728,
                5.59531526444068827,
                5.43444479314453499,
                5.07517453535834395,
            ]
        )
        f, g, h = g_suite.g22(x)
        assert f == pytest.approx(236.430975504001, abs=1e-3)
        assert np.all(g <= 1e-2)
        assert np.all(np.abs(h) <= 1e-2)


class TestG23:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g23(np.ones(8))

    def test_known_optimum_objective_and_feasibility(self):
        # g23's optimal vector as printed in the source PDF is corrupted by
        # a text-extraction artifact (its 4th equality constraint doesn't
        # check out against it). This point was reconstructed from the
        # other three equality constraints of the same problem and verified
        # independently to land within 0.002 of the documented optimum.
        x = np.array([0.0051, 99.9947, 0.0, 99.9999, 0.0001, 0.0, 100.0, 199.9999, 0.01])
        f, g, h = g_suite.g23(x)
        assert f == pytest.approx(-400.0551, abs=5e-3)
        assert np.all(g <= 1e-2)
        assert np.all(np.abs(h) <= 1e-2)


class TestG24:
    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            g_suite.g24(np.ones(3))

    def test_return_shape(self):
        f, g, h = g_suite.g24(np.ones(2))
        assert isinstance(f, float)
        assert isinstance(g, np.ndarray) and g.shape == (2,)
        assert isinstance(h, np.ndarray) and h.shape == (0,)

    def test_known_optimum_objective_and_feasibility(self):
        x = np.array([2.329520197, 3.178493496])
        f, g, h = g_suite.g24(x)
        assert f == pytest.approx(-5.508013271, abs=1e-3)
        assert np.all(g <= 1e-2)


@pytest.mark.parametrize(
    "func,x",
    [
        (g_suite.g01, np.ones(13)),
        (g_suite.g02, np.full(20, 0.5)),
        (g_suite.g03, np.full(10, 0.5)),
        (g_suite.g04, np.ones(5)),
        (g_suite.g05, np.ones(4)),
        (g_suite.g06, np.ones(2)),
        (g_suite.g07, np.ones(10)),
        (g_suite.g08, np.ones(2)),
        (g_suite.g09, np.ones(7)),
        (g_suite.g10, np.ones(8)),
        (g_suite.g11, np.ones(2)),
        (g_suite.g12, np.ones(3)),
        (g_suite.g13, np.ones(5)),
        (g_suite.g14, np.ones(10)),
        (g_suite.g15, np.ones(3)),
        (g_suite.g16, np.array([705.0, 68.6, 102.9, 282.3, 37.6])),
        (g_suite.g17, np.ones(6)),
        (g_suite.g18, np.ones(9)),
        (g_suite.g19, np.ones(15)),
        (g_suite.g20, np.ones(24)),
        (g_suite.g21, np.ones(7)),
        # g22 needs x10 > 100 (h12 takes log(x10 - 100)); a plain ones()
        # vector is out of that constraint's domain.
        (g_suite.g22, np.array([1.0] * 9 + [150.0] + [1.0] * 12)),
        (g_suite.g23, np.ones(9)),
        (g_suite.g24, np.ones(2)),
    ],
)
def test_return_types_are_consistent_across_functions(func, x):
    f, g, h = func(x)
    assert isinstance(f, float)
    assert isinstance(g, np.ndarray)
    assert isinstance(h, np.ndarray)


def test_g_suite_module_exposed_on_package():
    from bananabench import g_suite as pkg_g_suite

    assert pkg_g_suite.g04 is g_suite.g04


class TestGSuiteRegistry:
    def test_covers_all_24_problems(self):
        assert set(g_suite.G_SUITE.keys()) == {f"g{i:02d}" for i in range(1, 25)}

    def test_get_g_function_list_matches_registry(self):
        assert set(g_suite.get_g_function_list()) == set(g_suite.G_SUITE.keys())

    @pytest.mark.parametrize("name", list(g_suite.G_SUITE.keys()))
    def test_entry_function_matches_module_function(self, name):
        assert g_suite.G_SUITE[name]["function"] is getattr(g_suite, name)

    @pytest.mark.parametrize("name", list(g_suite.G_SUITE.keys()))
    def test_constraint_counts_match_actual_returned_arrays(self, name):
        # Cross-check the registry's n_inequality/n_equality against a live call,
        # not just the transcribed docstring, so the two can't silently drift.
        entry = g_suite.G_SUITE[name]
        x = np.ones(entry["dim"])
        f, g, h = entry["function"](x)
        assert len(g) == entry["n_inequality"]
        assert len(h) == entry["n_equality"]
