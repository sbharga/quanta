from __future__ import annotations

from pathlib import Path

from quanta import create_app


def test_home_and_session_api(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    app.extensions["quanta"]["runs"].start = lambda *_args, **_kwargs: True
    client = app.test_client()

    assert client.get("/").status_code == 200
    invalid = client.post("/api/sessions", json={"dataset": "not-a-slug"})
    assert invalid.status_code == 400

    created = client.post("/api/sessions", json={"dataset": "owner/dataset"})
    assert created.status_code == 201
    session_id = created.get_json()["id"]
    restored = client.get(f"/api/sessions/{session_id}")
    assert restored.status_code == 200
    assert restored.get_json()["dataset_ref"] == "owner/dataset"


def test_rename_and_delete_session(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    app.extensions["quanta"]["runs"].start = lambda *_args, **_kwargs: True
    client = app.test_client()

    created = client.post("/api/sessions", json={"dataset": "owner/dataset"})
    session_id = created.get_json()["id"]

    empty_title = client.patch(f"/api/sessions/{session_id}", json={"title": "   "})
    assert empty_title.status_code == 400

    renamed = client.patch(f"/api/sessions/{session_id}", json={"title": "My analysis"})
    assert renamed.status_code == 200
    assert renamed.get_json()["title"] == "My analysis"
    assert client.get(f"/api/sessions/{session_id}").get_json()["title"] == "My analysis"

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/sessions/{session_id}").status_code == 404
    assert client.delete(f"/api/sessions/{session_id}").status_code == 404


def test_notebook_download_headers_and_content(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    app.extensions["quanta"]["runs"].start = lambda *_args, **_kwargs: True
    client = app.test_client()

    created = client.post("/api/sessions", json={"dataset": "owner/dataset"})
    session_id = created.get_json()["id"]
    notebook = app.extensions["quanta"]["notebook"]
    notebook.create_cell(session_id, "markdown", "# Notes")

    response = client.get(f"/api/sessions/{session_id}/notebook.ipynb")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert session_id[:8] in response.headers["Content-Disposition"]
    document = response.get_json()
    assert document["nbformat"] == 4
    assert document["cells"][0]["cell_type"] == "markdown"

    assert client.get("/api/sessions/does-not-exist/notebook.ipynb").status_code == 404


def test_get_session_includes_timeline(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    app.extensions["quanta"]["runs"].start = lambda *_args, **_kwargs: True
    client = app.test_client()

    created = client.post("/api/sessions", json={"dataset": "owner/dataset"})
    session_id = created.get_json()["id"]
    app.extensions["quanta"]["events"].emit(session_id, "user_request", content="Explore")

    body = client.get(f"/api/sessions/{session_id}").get_json()
    assert "timeline" in body
    assert body["event_cursor"] >= 1
    assert any(entry["type"] == "request" for entry in body["timeline"])


def test_settings_api_masks_and_persists_tokens(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    client = app.test_client()

    updated = client.put(
        "/api/settings",
        json={
            "llm_api_key": "llm-secret",
            "kaggle_api_token": "kg-secret",
            "llm_model": "anthropic/claude-sonnet-4",
            "llm_base_url": "https://api.openai.com/v1",
        },
    )

    assert updated.status_code == 200
    body = updated.get_json()
    assert body["llm_configured"] is True
    assert body["llm_base_url"] == "https://api.openai.com/v1"
    assert body["kaggle_configured"] is True
    assert "llm-secret" not in updated.get_data(as_text=True)
    assert "kg-secret" not in client.get("/api/settings").get_data(as_text=True)
    assert (tmp_path / "settings.json").exists()


def test_settings_api_rejects_invalid_base_url(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.put("/api/settings", json={"llm_base_url": "not-a-url"})

    assert response.status_code == 400


def test_stopped_session_uses_composer_as_a_new_followup(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    store = app.extensions["quanta"]["store"]
    session = store.create_session("owner/dataset")
    store.update_session(session["id"], status="interrupted")
    calls = []
    app.extensions["quanta"]["runs"].start = lambda *args, **kwargs: calls.append((args, kwargs)) or True

    response = app.test_client().post(
        f"/api/sessions/{session['id']}/responses",
        json={"content": "Continue with the existing analysis"},
    )

    assert response.status_code == 200
    assert calls == [((session["id"],), {"followup": "Continue with the existing analysis"})]
