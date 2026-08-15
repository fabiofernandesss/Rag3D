import inspect
import math
from collections.abc import Mapping

import pytest

from rag3d.backend import DEFAULT_RETRIEVAL_LIMITS
from rag3d.config import TriRagConfig
import rag3d.fusion as fusion_module
from rag3d.fusion import fuse, quantum_fuse, rrf_fuse


def test_rrf_uses_one_based_ranks_and_absent_documents_contribute_zero():
    hits = rrf_fuse(
        {
            "semantico": [(2, 0.9), (1, 0.8)],
            "lexico": [(1, 100.0)],
        },
        {"semantico": 1.0, "lexico": 1.0},
        top_k=2,
        rrf_k=60,
    )

    assert [hit.chunk_id for hit in hits] == [1, 2]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[1].score == pytest.approx(1 / 61)


def test_rrf_counts_a_duplicate_only_at_its_first_rank():
    hits = rrf_fuse(
        {"semantico": [(1, 3.0), (1, 2.0), (2, 1.0)]},
        {"semantico": 1.0},
        top_k=2,
        rrf_k=60,
    )

    by_id = {hit.chunk_id: hit for hit in hits}
    assert by_id[1].score == pytest.approx(1 / 61)
    assert by_id[1].channels == ["semantico"]
    assert by_id[2].score == pytest.approx(1 / 63)


def test_rrf_ties_are_deterministic_by_chunk_id():
    channels = {
        "semantico": [(9, 1.0), (2, 0.5)],
        "lexico": [(2, 1.0), (9, 0.5)],
    }

    first = rrf_fuse(channels, {"semantico": 1.0, "lexico": 1.0}, 2)
    second = rrf_fuse(channels, {"semantico": 1.0, "lexico": 1.0}, 2)

    assert [hit.chunk_id for hit in first] == [2, 9]
    assert first == second


def test_top_k_zero_is_empty_for_both_fusion_methods():
    channels = {"semantico": [(1, 1.0)]}

    assert rrf_fuse(channels, {"semantico": 1.0}, 0) == []
    assert quantum_fuse(channels, {"semantico": 1.0}, 0) == []


@pytest.mark.parametrize("top_k", [-1, DEFAULT_RETRIEVAL_LIMITS.max_pool + 1])
def test_fusion_rejects_invalid_pool_limits(top_k):
    with pytest.raises(ValueError, match="top_k"):
        rrf_fuse({}, {}, top_k)


def test_fusion_rejects_an_honestly_oversized_channel_before_iteration():
    class ExplodingRanking:
        def __len__(self):
            return DEFAULT_RETRIEVAL_LIMITS.max_channel_k + 1

        def __iter__(self):
            raise AssertionError("oversized rankings must fail before iteration")

    with pytest.raises(ValueError, match="ranking.*maximum"):
        rrf_fuse({"semantico": ExplodingRanking()}, {"semantico": 1.0}, 1)


def test_fusion_stops_a_lying_channel_at_the_first_item_above_the_bound():
    class LyingRanking:
        def __init__(self):
            self.consumed = 0

        def __len__(self):
            return 0

        def __iter__(self):
            for chunk_id in range(DEFAULT_RETRIEVAL_LIMITS.max_channel_k + 1):
                self.consumed += 1
                yield (chunk_id, 1.0)

    ranking = LyingRanking()
    with pytest.raises(ValueError, match="ranking.*maximum"):
        rrf_fuse({"semantico": ranking}, {"semantico": 1.0}, 1)

    assert ranking.consumed == DEFAULT_RETRIEVAL_LIMITS.max_channel_k + 1


def test_fusion_rejects_too_many_channels_before_reading_rankings():
    class ExplodingRanking:
        def __iter__(self):
            raise AssertionError("channel-count preflight must run first")

    channels = {
        f"channel-{index}": ExplodingRanking()
        for index in range(DEFAULT_RETRIEVAL_LIMITS.max_fusion_channels + 1)
    }

    with pytest.raises(ValueError, match="channels.*maximum"):
        rrf_fuse(channels, {}, 1)


def test_fusion_counts_channels_from_a_lying_mapping():
    class LyingChannels(Mapping):
        def __init__(self):
            self.consumed = 0

        def __len__(self):
            return 0

        def __iter__(self):
            for index in range(
                DEFAULT_RETRIEVAL_LIMITS.max_fusion_channels + 1
            ):
                self.consumed += 1
                yield f"channel-{index}"

        def __getitem__(self, _key):
            return [(1, 1.0)]

    channels = LyingChannels()

    with pytest.raises(ValueError, match="channels.*maximum"):
        rrf_fuse(channels, {}, 1)

    assert channels.consumed == DEFAULT_RETRIEVAL_LIMITS.max_fusion_channels + 1


def test_fusion_rejects_too_many_weight_entries():
    weights = {
        f"channel-{index}": 1.0
        for index in range(DEFAULT_RETRIEVAL_LIMITS.max_fusion_channels + 1)
    }

    with pytest.raises(ValueError, match="weights.*maximum"):
        rrf_fuse({"semantic": [(1, 1.0)]}, weights, 1)


@pytest.mark.parametrize(
    "rrf_k",
    [0, -1, True, DEFAULT_RETRIEVAL_LIMITS.max_rrf_k + 1],
)
def test_rrf_rejects_invalid_rank_constant(rrf_k):
    error = (TypeError, ValueError)
    with pytest.raises(error, match="rrf_k"):
        rrf_fuse({}, {}, 1, rrf_k=rrf_k)


@pytest.mark.parametrize("weight", [-0.1, math.inf, math.nan])
def test_rrf_rejects_unsafe_weights_without_emitting_nan(weight):
    with pytest.raises(ValueError, match="weight"):
        rrf_fuse({"semantico": [(1, 1.0)]}, {"semantico": weight}, 1)


def test_rrf_rejects_finite_inputs_when_the_accumulated_score_overflows():
    channels = {
        f"channel-{index}": [(1, 1.0)]
        for index in range(DEFAULT_RETRIEVAL_LIMITS.max_fusion_channels)
    }
    weights = {name: 1e308 for name in channels}

    with pytest.raises(ValueError, match="non-finite"):
        rrf_fuse(channels, weights, 1, rrf_k=1)


def test_nonfinite_channel_scores_fail_closed():
    with pytest.raises(ValueError, match="score"):
        quantum_fuse(
            {"semantico": [(1, math.nan)]},
            {"semantico": 1.0},
            top_k=1,
        )


def test_negative_finite_scores_do_not_make_quantum_results_nonfinite():
    hits = quantum_fuse(
        {"semantico": [(1, -3.0), (2, -9.0)]},
        {"semantico": 1.0},
        top_k=2,
    )

    assert all(math.isfinite(hit.score) for hit in hits)
    assert all(math.isfinite(hit.classical) for hit in hits)
    assert all(math.isfinite(hit.interference) for hit in hits)


def test_quantum_extreme_finite_scores_stay_finite():
    hits = quantum_fuse(
        {"semantico": [(1, 1e308), (2, -1e308)]},
        {"semantico": 1.0},
        top_k=2,
    )

    assert [hit.chunk_id for hit in hits] == [1, 2]
    assert all(math.isfinite(hit.score) for hit in hits)
    assert all(math.isfinite(hit.classical) for hit in hits)
    assert all(math.isfinite(hit.interference) for hit in hits)
    assert all(
        math.isfinite(score)
        for hit in hits
        for score in hit.per_channel.values()
    )


def test_quantum_near_flat_channel_preserves_legacy_cross_language_tie_rule():
    hits = quantum_fuse(
        {"semantico": [(2, 1.0), (1, 1.0 - 5e-13)]},
        {"semantico": 1.0},
        top_k=2,
    )

    assert [hit.chunk_id for hit in hits] == [1, 2]
    assert [hit.score for hit in hits] == [1.0, 1.0]


def test_quantum_rejects_finite_inputs_when_the_combined_score_overflows():
    channels = {
        "semantico": [(1, 2.0), (2, 1.0)],
        "lexico": [(1, 2.0), (2, 1.0)],
    }

    with pytest.raises(ValueError, match="non-finite"):
        quantum_fuse(
            channels,
            {"semantico": 1e308, "lexico": 1e308},
            top_k=1,
        )


def test_quantum_preserves_compatibility_fields():
    hit = quantum_fuse(
        {
            "semantico": [(1, 1.0), (2, 0.0)],
            "lexico": [(1, 2.0)],
        },
        {"semantico": 1.0, "lexico": 1.0},
        top_k=1,
    )[0]

    assert hit.chunk_id == 1
    assert math.isfinite(hit.classical)
    assert math.isfinite(hit.interference)
    assert hit.channels == ["semantico", "lexico"]
    assert hit.per_channel == {"semantico": 1.0, "lexico": 1.0}


def test_fuse_rejects_unknown_method_and_ambiguous_weight_count():
    channels = {"semantico": [(1, 1.0)], "lexico": [(2, 1.0)]}

    with pytest.raises(ValueError, match="method"):
        fuse(channels, (1.0, 1.0), 2, method="typo")
    with pytest.raises(ValueError, match="weights"):
        fuse(channels, (1.0,), 2, method="rrf")


def test_legacy_fuse_keeps_explicit_rrf_coherence_weighting_compatible():
    channels = {
        "decisivo": [(1, 1.0), (2, 0.5), (3, 0.2), (4, 0.0)],
        "flat": [(4, 1.0), (3, 1.0), (2, 1.0), (1, 1.0)],
    }

    legacy = fuse(
        channels,
        (1.0, 1.0),
        4,
        method="rrf",
        coherence_strength=1.0,
    )
    expected = rrf_fuse(channels, {"decisivo": 1.0, "flat": 0.0}, 4)

    assert [(hit.chunk_id, hit.score) for hit in legacy] == pytest.approx(
        [(hit.chunk_id, hit.score) for hit in expected]
    )


def test_greedy_dpp_documentation_does_not_claim_exact_global_map():
    fusion_source = inspect.getsource(fusion_module)
    config_source = inspect.getsource(TriRagConfig)

    assert "MAP-DPP" not in fusion_source
    assert "MAP-DPP" not in config_source
    assert "argmax é obtido pelo guloso" not in fusion_source


try:
    from hypothesis import given, strategies as st
except ImportError:  # optional test dependency
    given = None


if given is not None:

    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=64),
            min_size=1,
            max_size=20,
        )
    )
    def test_quantum_property_finite_scores_return_finite_or_fail_explicitly(scores):
        ranking = [(index, score) for index, score in enumerate(scores)]

        try:
            hits = quantum_fuse(
                {"semantico": ranking},
                {"semantico": 1.0},
                len(ranking),
            )
        except ValueError as exc:
            assert "quantum" in str(exc).lower()
        else:
            assert all(math.isfinite(hit.score) for hit in hits)
            assert all(math.isfinite(hit.classical) for hit in hits)
            assert all(math.isfinite(hit.interference) for hit in hits)

    @given(st.lists(st.integers(min_value=0, max_value=100), unique=True, max_size=30))
    def test_rrf_property_is_deterministic_and_finite(ids):
        ranking = [(cid, float(len(ids) - rank)) for rank, cid in enumerate(ids)]
        channels = {"semantico": ranking, "lexico": list(reversed(ranking))}
        weights = {"semantico": 1.0, "lexico": 1.0}

        first = rrf_fuse(channels, weights, len(ids))
        second = rrf_fuse(channels, weights, len(ids))

        assert first == second
        assert all(math.isfinite(hit.score) for hit in first)

    @given(
        st.lists(st.integers(min_value=0, max_value=1000), unique=True, max_size=40),
        st.integers(min_value=0, max_value=40),
    )
    def test_rrf_property_absent_is_zero_and_top_k_is_bounded(ids, requested):
        ranking = [(cid, -float(rank)) for rank, cid in enumerate(ids)]
        hits = rrf_fuse(
            {"semantico": ranking, "lexico": []},
            {"semantico": 1.0, "lexico": 1.0},
            requested,
            rrf_k=60,
        )

        assert len(hits) <= min(requested, len(ids))
        original_rank = {cid: rank + 1 for rank, cid in enumerate(ids)}
        for hit in hits:
            assert hit.score == pytest.approx(1.0 / (60 + original_rank[hit.chunk_id]))
            assert hit.channels == ["semantico"]
