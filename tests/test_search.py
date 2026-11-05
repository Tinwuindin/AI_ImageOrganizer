from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.context import AppContext
from app.database.models import STATUS_MISSING, Classification, ScannedFile
from app.services.jobs import JobContext, JobRunner
from app.services.search_service import (
    build_match_expression,
    extract_terms,
    reciprocal_rank_fusion,
)
from tests.conftest import FakeClassifier, FakeImageEncoder, FakeQueryEncoder, make_image

Fakes = tuple[FakeImageEncoder, FakeQueryEncoder, FakeClassifier]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("fotos de mi perro en la playa", ["perro", "playa"]),
        ("Muéstrame una imagen del cumpleaños", ["cumpleaños"]),
        ("gato gato GATO", ["gato"]),
        ("de la el", []),
        ("", []),
    ],
)
def test_extract_terms(query: str, expected: list[str]) -> None:
    assert extract_terms(query) == expected


def test_match_expression_covers_plurals_and_escapes_syntax() -> None:
    assert build_match_expression(["perros", "sol"]) == '"perro"* OR "sol"*'


def test_rrf_rewards_appearing_in_both_rankings() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4]], k=60)

    assert fused[0] == 3
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_handles_an_empty_ranking() -> None:
    assert reciprocal_rank_fusion([[5, 6], []], k=60) == [5, 6]


@pytest.fixture
def library(context: AppContext, tmp_path: Path) -> dict[str, int]:
    root = tmp_path / "fotos"
    colors = {"tomate": (230, 20, 20), "bosque": (20, 200, 20), "mar": (20, 20, 230)}
    for name, color in colors.items():
        make_image(root / f"{name}.png", color, (32, 32))
    context.pipeline.run(JobContext(JobRunner()), [root])
    ids = {
        record.filename.removesuffix(".png"): record.id
        for record in context.repository.all_images().values()
    }
    context.repository.save_classifications([
        Classification(
            ids["mar"], "Paisajes y naturaleza", "Playa",
            description="Atardecer frente al mar con un velero",
            tags=("playa", "cumpleaños"),
            ocr_text="Bienvenidos a Mazatlán",
            confidence=0.9,
        ),
    ])
    return ids


def test_semantic_search_finds_by_meaning(context: AppContext, library: dict[str, int]) -> None:
    result = context.search.search("algo rojo")

    assert result["semantic"] is True
    assert result["results"][0]["id"] == library["tomate"]
    assert result["results"][0]["similarity"] is not None


@pytest.mark.parametrize(
    "query",
    ["velero", "VELERO", "playa", "playas", "cumpleanos", "Cumpleaños", "Mazatlán", "mazatlan"],
)
def test_text_search_reads_description_tags_and_visible_text(
    context: AppContext,
    library: dict[str, int],
    query: str,
) -> None:
    result = context.search.search(query)

    assert result["results"][0]["id"] == library["mar"]
    assert result["results"][0]["text_match"] is True


def test_finding_by_filename(context: AppContext, library: dict[str, int]) -> None:
    result = context.search.search("tomate")

    assert result["results"][0]["id"] == library["tomate"]


def test_text_search_works_without_the_encoder(
    context: AppContext,
    library: dict[str, int],
    fakes: Fakes,
) -> None:
    fakes[1].fail = True

    result = context.search.search("velero")

    assert result["semantic"] is False
    assert [item["id"] for item in result["results"]] == [library["mar"]]


def test_category_filter(context: AppContext, library: dict[str, int]) -> None:
    result = context.search.search("algo rojo", category="Paisajes y naturaleza")

    assert {item["category"] for item in result["results"]} == {"Paisajes y naturaleza"}


def test_limit(context: AppContext, library: dict[str, int]) -> None:
    assert len(context.search.search("rojo", limit=2)["results"]) == 2


def test_missing_images_are_not_returned(context: AppContext, library: dict[str, int]) -> None:
    context.repository.set_status([library["tomate"]], STATUS_MISSING)

    result = context.search.search("algo rojo")

    assert library["tomate"] not in {item["id"] for item in result["results"]}


@pytest.mark.parametrize("query", ['"', "(", "AND OR NOT", "a* b:c", "-", "'; DROP TABLE images"])
def test_special_characters_do_not_break_the_search(
    context: AppContext,
    library: dict[str, int],
    query: str,
) -> None:
    assert "results" in context.search.search(query)


def test_empty_library(context: AppContext) -> None:
    assert context.search.search("lo que sea") == {
        "query": "lo que sea", "semantic": False, "results": [],
    }


def test_a_dominant_topic_does_not_empty_the_results(context: AppContext) -> None:
    generator = np.random.default_rng(0)
    count = 90
    ids = [
        context.repository.insert_image(
            ScannedFile(f"C:/fotos/{index}.jpg", f"hash{index}", 10, 1.0, 10, 10)
        )
        for index in range(count)
    ]
    # Un tercio de la biblioteca trata de lo que se busca.
    vectors = generator.normal(0, 0.05, (count, 16)).astype(np.float32)
    vectors[:, 0] += [1.0 if index < count // 3 else 0.2 for index in range(count)]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    context.vectors.add(ids, vectors)

    results = context.search.search("algo rojo", limit=30)["results"]

    assert {item["id"] for item in results} == set(ids[:30])
