from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.context import AppContext
from app.database.models import OP_DONE
from app.main import create_app
from tests.conftest import make_image

BASE_URL = "http://127.0.0.1:8765"


@pytest.fixture
def client(context: AppContext) -> Iterator[TestClient]:
    with TestClient(create_app(context.config, context), base_url=BASE_URL) as test_client:
        yield test_client


def wait_for_job(client: TestClient, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/pipeline/status").json()
        if status["state"] not in ("running", "cancelling"):
            return status
        time.sleep(0.02)
    raise AssertionError("La tarea no terminó a tiempo.")


@pytest.fixture
def library(client: TestClient, tmp_path: Path) -> Path:
    root = tmp_path / "fotos"
    for name, color in (("rojo", (220, 30, 30)), ("verde", (30, 200, 30))):
        for index in range(6):
            make_image(root / f"{name}_{index}.png", color, (48, 48), seed=index)
    response = client.post("/api/pipeline/start", json={"roots": [str(root)]})
    assert response.status_code == 202
    assert wait_for_job(client)["state"] == "completed"
    return root


def first_image(client: TestClient) -> dict:
    return client.get("/api/images", params={"page_size": 1}).json()["items"][0]


def test_home_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_analysis_rejects_missing_folders(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/api/pipeline/start", json={"roots": [str(tmp_path / "no-existe")]})

    assert response.status_code == 422


def test_analysis_rejects_relative_paths(client: TestClient) -> None:
    response = client.post("/api/pipeline/start", json={"roots": ["fotos"]})

    assert response.status_code == 422


def test_listing_and_filters(client: TestClient, library: Path) -> None:
    listing = client.get("/api/images").json()
    assert listing["total"] == 12

    filtered = client.get("/api/images", params={"category": "Comida y bebida"}).json()
    assert filtered["total"] == 6
    assert {item["category"] for item in filtered["items"]} == {"Comida y bebida"}


def test_pagination(client: TestClient, library: Path) -> None:
    first = client.get("/api/images", params={"page": 1, "page_size": 5}).json()["items"]
    second = client.get("/api/images", params={"page": 2, "page_size": 5}).json()["items"]

    assert not {item["id"] for item in first} & {item["id"] for item in second}


@pytest.mark.parametrize("sort", ["date_desc", "date_asc", "name", "size", "confidence", "raro"])
def test_sort_options(client: TestClient, library: Path, sort: str) -> None:
    assert client.get("/api/images", params={"sort": sort}).status_code == 200


def test_image_detail_thumbnail_and_preview(client: TestClient, library: Path) -> None:
    image = first_image(client)

    detail = client.get(f"/api/images/{image['id']}")
    thumbnail = client.get(f"/api/images/{image['id']}/thumbnail", params={"v": "abc"})
    unversioned = client.get(f"/api/images/{image['id']}/thumbnail")
    preview = client.get(f"/api/images/{image['id']}/preview")

    assert detail.json()["filename"] == image["filename"]
    assert thumbnail.headers["content-type"] == "image/jpeg"
    assert "immutable" in thumbnail.headers["cache-control"]
    assert "no-cache" in unversioned.headers["cache-control"]
    assert preview.status_code == 200
    assert preview.content[:2] == b"\xff\xd8"


def test_unknown_image_is_a_404(client: TestClient) -> None:
    assert client.get("/api/images/999").status_code == 404
    assert client.get("/api/images/999/thumbnail").status_code == 404
    assert client.get("/api/images/999/preview").status_code == 404


def test_similar_images_exclude_the_image_itself(client: TestClient, library: Path) -> None:
    image = first_image(client)

    results = client.get(f"/api/images/{image['id']}/similar").json()["results"]

    assert results
    assert image["id"] not in {item["id"] for item in results}
    assert results[0]["category"] == image["category"]


def test_manual_classification(client: TestClient, library: Path) -> None:
    image = first_image(client)

    response = client.patch(
        f"/api/images/{image['id']}/classification",
        json={"category": "Documentos", "subcategory": "Facturas"},
    )

    assert response.status_code == 200
    assert response.json()["source"] == "manual"
    assert response.json()["confidence"] == 1.0


def test_manual_classification_requires_an_existing_category(
    client: TestClient, library: Path
) -> None:
    image = first_image(client)

    response = client.patch(
        f"/api/images/{image['id']}/classification", json={"category": "Inventada"}
    )

    assert response.status_code == 422


def test_search_endpoint(client: TestClient, library: Path) -> None:
    response = client.get("/api/search", params={"q": "rojo"})

    assert response.status_code == 200
    assert response.json()["results"]
    assert client.get("/api/search").status_code == 422


def test_category_lifecycle(client: TestClient, library: Path) -> None:
    created = client.post(
        "/api/categories", json={"name": "Facturas", "description": "Recibos de compra"}
    )
    assert created.status_code == 201
    category_id = created.json()["id"]

    tree = client.get("/api/categories").json()["categories"]
    assert tree[0]["name"] == "Facturas", "Las categorías propias van primero."

    assert client.post("/api/categories", json={"name": "facturas"}).status_code == 409

    renamed = client.patch(f"/api/categories/{category_id}", json={"name": "Tickets"})
    assert renamed.status_code == 200

    assert client.delete(f"/api/categories/{category_id}").status_code == 200
    names = {node["name"] for node in client.get("/api/categories").json()["categories"]}
    assert "Tickets" not in names


def test_renaming_a_category_updates_its_images_and_search(
    client: TestClient, library: Path
) -> None:
    tree = client.get("/api/categories").json()["categories"]
    node = next(item for item in tree if item["name"] == "Comida y bebida")

    client.patch(f"/api/categories/{node['id']}", json={"name": "Gastronomía"})

    filtered = client.get("/api/images", params={"category": "Gastronomía"}).json()
    assert filtered["total"] == 6
    assert client.get("/api/search", params={"q": "gastronomia"}).json()["results"]


def test_deleting_a_category_reassigns_its_images(client: TestClient, library: Path) -> None:
    tree = client.get("/api/categories").json()["categories"]
    node = next(item for item in tree if item["name"] == "Comida y bebida")

    response = client.delete(f"/api/categories/{node['id']}")

    assert response.status_code == 200
    assert client.get("/api/images", params={"category": "Otros"}).json()["total"] == 6


def test_fallback_category_is_protected(client: TestClient) -> None:
    tree = client.get("/api/categories").json()["categories"]
    fallback = next(item for item in tree if item["name"] == "Otros")

    assert client.delete(f"/api/categories/{fallback['id']}").status_code == 409
    assert client.patch(
        f"/api/categories/{fallback['id']}", json={"name": "Varios"}
    ).status_code == 409


def test_organizing_flow_with_undo(
    client: TestClient, library: Path, tmp_path: Path, context: AppContext
) -> None:
    destination = tmp_path / "ordenadas"

    proposal = client.post("/api/organizer/plans", json={"destination": str(destination)})

    assert proposal.status_code == 201
    plan = proposal.json()
    assert plan["status"] == "draft"
    assert plan["total"] == 12
    assert not destination.exists(), "Proponer no debe tocar el disco."
    assert plan["folders"][0]["sample_ids"]

    excluded = plan["folders"][0]
    client.patch(
        f"/api/organizer/plans/{plan['id']}/folders",
        json={"folder": excluded["folder"], "included": False},
    )

    assert client.post(f"/api/organizer/plans/{plan['id']}/execute").status_code == 202
    assert wait_for_job(client)["summary"]["moved"] == 12 - excluded["total"]
    moved = list(destination.rglob("*.png"))
    assert len(moved) == 12 - excluded["total"]
    assert client.post(f"/api/organizer/plans/{plan['id']}/execute").status_code == 409

    assert client.post(f"/api/organizer/plans/{plan['id']}/undo").status_code == 202
    assert wait_for_job(client)["summary"]["restored"] == 12 - excluded["total"]
    assert len(list(library.glob("*.png"))) == 12
    assert list(destination.rglob("*.png")) == []
    assert all(
        operation.status != OP_DONE
        for operation in context.repository.plan_operations(plan["id"])
    )


def test_plan_rejects_bad_input(client: TestClient, library: Path, tmp_path: Path) -> None:
    relative = client.post("/api/organizer/plans", json={"destination": "fotos"})
    template = client.post(
        "/api/organizer/plans",
        json={"destination": str(tmp_path / "x"), "template": "../{category}"},
    )

    assert relative.status_code == 422
    assert template.status_code == 422


def test_operations_of_a_folder_can_be_toggled(
    client: TestClient, library: Path, tmp_path: Path
) -> None:
    plan = client.post(
        "/api/organizer/plans", json={"destination": str(tmp_path / "ordenadas")}
    ).json()
    folder = plan["folders"][0]["folder"]

    operations = client.get(
        f"/api/organizer/plans/{plan['id']}/operations", params={"folder": folder}
    ).json()["operations"]
    client.patch(f"/api/organizer/operations/{operations[0]['id']}", json={"included": False})

    detail = client.get(f"/api/organizer/plans/{plan['id']}").json()
    assert detail["included"] == detail["total"] - 1


def test_draft_can_be_discarded_but_not_an_executed_plan(
    client: TestClient, library: Path, tmp_path: Path
) -> None:
    plan = client.post(
        "/api/organizer/plans", json={"destination": str(tmp_path / "ordenadas")}
    ).json()
    client.post(f"/api/organizer/plans/{plan['id']}/execute")
    wait_for_job(client)

    assert client.delete(f"/api/organizer/plans/{plan['id']}").status_code == 409


def test_only_one_background_job_at_a_time(
    client: TestClient, tmp_path: Path, context: AppContext
) -> None:
    root = tmp_path / "fotos"
    make_image(root / "a.png")
    client.post("/api/pipeline/start", json={"roots": [str(root)]})

    second = client.post("/api/pipeline/start", json={"roots": [str(root)]})

    assert second.status_code in (202, 409)
    wait_for_job(client)


def test_requirements_report_missing_models(client: TestClient) -> None:
    body = client.get("/api/system/info").json()

    assert body["ready"] is False
    assert "Gemma 4 E2B" in body["missing"]
    assert body["min_confidence"] == 0.8


@pytest.mark.parametrize(
    "host", ["evil.example", "127.0.0.1.evil.example", "localhost.evil.example"]
)
def test_dns_rebinding_is_blocked(client: TestClient, host: str) -> None:
    response = client.get("/api/images", headers={"Host": host})

    assert response.status_code == 403


def test_other_websites_cannot_send_commands(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/organizer/plans",
        json={"destination": str(tmp_path)},
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 403


def test_the_local_interface_can_send_commands(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/organizer/plans",
        json={"destination": str(tmp_path / "x")},
        headers={"Origin": BASE_URL},
    )

    assert response.status_code == 201


def test_undo_requires_an_executed_plan(
    client: TestClient, library: Path, tmp_path: Path
) -> None:
    plan = client.post(
        "/api/organizer/plans", json={"destination": str(tmp_path / "ordenadas")}
    ).json()

    assert client.post(f"/api/organizer/plans/{plan['id']}/undo").status_code == 409


def test_copies_cannot_change_while_a_job_runs(
    client: TestClient, library: Path, context: AppContext
) -> None:
    release = threading.Event()
    context.jobs.start("prueba", lambda job: release.wait(5) and None)

    response = client.post("/api/images/duplicates/1/keep", json={"image_id": 1})

    release.set()
    context.jobs.wait(5)
    assert response.status_code == 409
