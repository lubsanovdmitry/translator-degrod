from __future__ import annotations

from itertools import pairwise

from semantic_telephone.models import RouteConfig
from semantic_telephone.routes import generate_route


def test_routes_are_seeded_and_end_at_target() -> None:
    config = RouteConfig(mode="random", min_hops=6, max_hops=6)
    first = generate_route(config, source_language="ru", target_language="ru", seed=17)
    second = generate_route(config, source_language="ru", target_language="ru", seed=17)
    assert first == second
    assert first[0] == "ru"
    assert first[-1] == "ru"
    assert all(left != right for left, right in pairwise(first))


def test_all_route_modes() -> None:
    for mode in ("fixed", "random", "stratified", "hubbed", "mutating_fixed"):
        config = RouteConfig(
            mode=mode,
            languages=["ru", "de", "en", "ru"],
            min_hops=5,
            max_hops=5,
        )
        route = generate_route(config, source_language="ru", target_language="ru", seed=9)
        assert route[-1] == "ru"
        assert all(a != b for a, b in pairwise(route))


def test_allow_and_deny_lists() -> None:
    config = RouteConfig(
        mode="random", allow=["de", "fr"], deny=["fr"], min_hops=3, max_hops=3
    )
    route = generate_route(config, source_language="ru", target_language="ru", seed=1)
    assert "fr" not in route


def test_deny_list_applies_to_fixed_and_hubbed_routes() -> None:
    fixed = RouteConfig(
        mode="fixed",
        languages=["ru", "de", "fr", "en"],
        deny=["de"],
    )
    fixed_route = generate_route(
        fixed, source_language="ru", target_language="en", seed=1
    )
    assert fixed_route == ["ru", "fr", "en"]

    hubbed = RouteConfig(
        mode="hubbed",
        deny=["en"],
        hub_language="en",
        min_hops=5,
        max_hops=5,
    )
    hubbed_route = generate_route(
        hubbed, source_language="ru", target_language="ru", seed=1
    )
    assert "en" not in hubbed_route
