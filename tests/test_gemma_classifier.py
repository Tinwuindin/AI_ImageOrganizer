from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest

from app.core.config import GemmaConfig
from app.database.models import Category
from app.models.gemma_classifier import (
    GemmaClassifier,
    GemmaError,
    ImageAnalysis,
    build_schema,
    build_system_prompt,
    category_probability,
    parse_analysis,
)
from tests.conftest import make_image

CATEGORIES = ["Vehículos", "Otros"]


def tokens_for(text: str, category_logprob: float = -0.1) -> list[dict]:
    """Parte el texto en tokens de un byte; los de la categoría llevan la logprob dada."""
    data = text.encode("utf-8")
    key = data.find(b'"category"')
    value = data.find(b"\xc3\xad", key)  # La í de Vehículos cae dentro del valor.
    tokens = []
    for index, byte in enumerate(data):
        logprob = category_logprob if index == value else 0.0
        tokens.append({"token": "?", "logprob": logprob, "bytes": [byte]})
    return tokens


RESPONSE = json.dumps({
    "description": "Un  auto rojo\nen la calle.",
    "text_in_image": " ABC 123 ",
    "tags": ["Auto", "auto", " Calle ", "rojo"],
    "subject": "coches clásicos",
    "category": "Vehículos",
}, ensure_ascii=False)


def test_schema_writes_the_topic_before_the_category() -> None:
    fields = list(build_schema(CATEGORIES)["properties"])

    assert fields.index("description") < fields.index("subject") < fields.index("category")
    assert "confidence" not in fields


def test_schema_limits_category_to_the_list() -> None:
    assert build_schema(CATEGORIES)["properties"]["category"]["enum"] == CATEGORIES


def test_system_prompt_lists_categories_in_order() -> None:
    prompt = build_system_prompt([
        Category(1, "Primera", "uno", 0),
        Category(2, "Segunda", "dos", 1),
    ])

    assert prompt.index("- Primera: uno") < prompt.index("- Segunda: dos")


def test_parse_normalizes_the_answer() -> None:
    analysis = parse_analysis(RESPONSE, CATEGORIES, tokens_for(RESPONSE))

    assert analysis.category == "Vehículos"
    assert analysis.subcategory == "Coches clásicos"
    assert analysis.description == "Un auto rojo en la calle."
    assert analysis.tags == ("auto", "calle", "rojo")
    assert analysis.ocr_text == "ABC 123"


def test_confidence_is_the_probability_of_the_category_tokens() -> None:
    tokens = tokens_for(RESPONSE, category_logprob=math.log(0.25))

    assert category_probability(tokens, "Vehículos") == pytest.approx(0.25)


def test_tokens_outside_the_category_do_not_affect_confidence() -> None:
    tokens = tokens_for(RESPONSE)
    tokens[3]["logprob"] = -5.0  # Pertenece a la descripción.

    assert category_probability(tokens, "Vehículos") == pytest.approx(math.exp(-0.1))


def test_token_spanning_the_start_of_the_value_counts() -> None:
    data = RESPONSE.encode("utf-8")
    value = "Vehículos".encode()
    start = data.find(value)
    end = start + len(value)
    tokens = [
        {"logprob": -1.0, "bytes": list(data[:start + 2])},  # Termina dentro del valor.
        {"logprob": -2.0, "bytes": list(data[start + 2:end])},
        {"logprob": -7.0, "bytes": list(data[end:])},  # Ya es parte del cierre del JSON.
    ]

    assert category_probability(tokens, "Vehículos") == pytest.approx(math.exp(-3.0))


def test_missing_logprobs_mean_no_confidence() -> None:
    assert category_probability([], "Vehículos") == 0.0


@pytest.mark.parametrize(
    "content",
    [
        "no es json",
        '{"category": "Vehículos"}',
        '{"category": "Inventada", "tags": []}',
        "[1, 2]",
    ],
)
def test_invalid_answers_are_errors(content: str) -> None:
    with pytest.raises(GemmaError):
        parse_analysis(content, CATEGORIES, [])


@pytest.fixture
def classifier(tmp_path: Path) -> GemmaClassifier:
    config = GemmaConfig(max_retries=2)
    return GemmaClassifier(config, tmp_path / "llama-server.exe", tmp_path / "llama.log")


def test_missing_binary_explains_how_to_fix_it(classifier: GemmaClassifier) -> None:
    with pytest.raises(GemmaError, match="download_models"):
        classifier.load()


def test_analyze_retries_after_a_transient_failure(
    classifier: GemmaClassifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = make_image(tmp_path / "a.jpg")
    answers = iter([httpx.ConnectError("caído"), (RESPONSE, tokens_for(RESPONSE))])

    def complete(_: dict) -> tuple[str, list[dict]]:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(classifier, "load", lambda: None)
    monkeypatch.setattr(classifier, "_complete", complete)

    analysis = classifier.analyze(image, [Category(1, "Vehículos", "", 0)])

    assert isinstance(analysis, ImageAnalysis)


def test_analyze_gives_up_after_the_retries(
    classifier: GemmaClassifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = make_image(tmp_path / "a.jpg")
    calls: list[int] = []

    def complete(_: dict) -> tuple[str, list[dict]]:
        calls.append(1)
        raise GemmaError("sin respuesta")

    monkeypatch.setattr(classifier, "load", lambda: None)
    monkeypatch.setattr(classifier, "_complete", complete)

    with pytest.raises(GemmaError, match="no pudo analizar"):
        classifier.analyze(image, [Category(1, "Vehículos", "", 0)])

    assert len(calls) == 3
