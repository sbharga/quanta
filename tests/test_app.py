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

    home = client.get("/")
    assert home.status_code == 200
    assert 'rel="icon"' in home.get_data(as_text=True)
    assert 'favicon.svg' in home.get_data(as_text=True)
    assert 'id="data-button"' in home.get_data(as_text=True)
    assert 'id="dataset-explorer"' in home.get_data(as_text=True)
    assert 'id="packages-dialog"' in home.get_data(as_text=True)
    invalid = client.post("/api/sessions", json={"dataset": "not-a-slug"})
    assert invalid.status_code == 400

    created = client.post("/api/sessions", json={"dataset": "owner/dataset"})
    assert created.status_code == 201
    session_id = created.get_json()["id"]
    restored = client.get(f"/api/sessions/{session_id}")
    assert restored.status_code == 200
    assert restored.get_json()["dataset_ref"] == "owner/dataset"


def test_session_api_exposes_dataset_manifest(tmp_path: Path) -> None:
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
    manifest = {
        "root": str(tmp_path / "data"),
        "file_count": 1,
        "total_bytes": 24,
        "errors": [],
        "files": [
            {
                "path": "scores.csv",
                "size": 24,
                "type": "csv",
                "inspection": {
                    "kind": "table",
                    "rows": 2,
                    "sample_rows": 2,
                    "columns": [
                        {
                            "name": "score",
                            "dtype": "int64",
                            "nulls_in_sample": 0,
                            "unique_in_sample": 2,
                        }
                    ],
                    "preview": [{"score": 10}, {"score": 20}],
                    "sheets": None,
                },
            }
        ],
    }
    store.update_session(session["id"], manifest=manifest)

    response = app.test_client().get(f"/api/sessions/{session['id']}")

    assert response.status_code == 200
    assert response.get_json()["manifest"] == manifest


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


def test_regenerate_route_starts_task_and_revises_cell(tmp_path: Path) -> None:
    app = create_app({"DATA_DIR": tmp_path, "DATABASE_PATH": tmp_path / "app.sqlite3", "LLM_API_KEY": None})
    app.config.update(TESTING=True)
    store = app.extensions["quanta"]["store"]
    session = store.create_session("owner/dataset")
    cell = store.create_cell(session["id"], "code", "old_code")
    captured = {}

    def fake_run_manual(session_id, task, message="Executing code"):
        captured["message"] = message
        task()
        return True

    app.extensions["quanta"]["runs"].run_manual = fake_run_manual
    app.extensions["quanta"]["engine"].regenerate_cell = lambda session_id, cell_id: store.revise_cell(
        cell_id, "new_code", "assistant"
    )
    app.extensions["quanta"]["notebook"].restore_kernel = lambda session_id: None
    app.extensions["quanta"]["notebook"].run_cell = lambda session_id, cell_id: None

    response = app.test_client().post(f"/api/sessions/{session['id']}/cells/{cell['id']}/regenerate")

    assert response.status_code == 202
    assert captured["message"] == "Regenerating cell"
    assert store.get_cell(cell["id"])["source"] == "new_code"


def test_regenerate_route_returns_409_when_session_is_busy(tmp_path: Path) -> None:
    app = create_app({"DATA_DIR": tmp_path, "DATABASE_PATH": tmp_path / "app.sqlite3", "LLM_API_KEY": None})
    app.config.update(TESTING=True)
    store = app.extensions["quanta"]["store"]
    session = store.create_session("owner/dataset")
    cell = store.create_cell(session["id"], "code", "old_code")
    app.extensions["quanta"]["runs"].run_manual = lambda *_args, **_kwargs: False

    response = app.test_client().post(f"/api/sessions/{session['id']}/cells/{cell['id']}/regenerate")

    assert response.status_code == 409


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


def test_settings_api_updates_and_clears_max_iterations(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
            "MAX_ITERATIONS": 12,
        }
    )
    app.config.update(TESTING=True)
    client = app.test_client()

    updated = client.put("/api/settings", json={"max_iterations": 4})
    assert updated.status_code == 200
    assert updated.get_json()["max_iterations"] == 4

    cleared = client.put("/api/settings", json={"max_iterations": None})
    assert cleared.status_code == 200
    assert cleared.get_json()["max_iterations"] == 12


def test_settings_api_rejects_out_of_range_max_iterations(tmp_path: Path) -> None:
    app = create_app({"DATA_DIR": tmp_path, "DATABASE_PATH": tmp_path / "app.sqlite3", "LLM_API_KEY": None})
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.put("/api/settings", json={"max_iterations": 0})

    assert response.status_code == 400


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


def test_packages_list_route_returns_recorded_packages(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)
    store = app.extensions["quanta"]["store"]
    store.record_package("seaborn", "seaborn==0.13.2", "0.13.2", "user", None)

    response = app.test_client().get("/api/packages")

    assert response.status_code == 200
    body = response.get_json()
    assert body["packages"] == [
        {
            "name": "seaborn",
            "requirement": "seaborn==0.13.2",
            "version": "0.13.2",
            "author": "user",
            "session_id": None,
            "installed_at": body["packages"][0]["installed_at"],
        }
    ]
    assert isinstance(body["backend_available"], bool)


def test_install_route_rejects_invalid_requirement(tmp_path: Path) -> None:
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

    response = app.test_client().post(
        f"/api/sessions/{session['id']}/packages", json={"specs": ["--index-url=https://evil.example"]}
    )

    assert response.status_code == 400


def test_install_route_returns_404_for_unknown_session(tmp_path: Path) -> None:
    app = create_app(
        {
            "DATA_DIR": tmp_path,
            "DATABASE_PATH": tmp_path / "app.sqlite3",
            "LLM_API_KEY": None,
        }
    )
    app.config.update(TESTING=True)

    response = app.test_client().post("/api/sessions/missing/packages", json={"specs": ["seaborn"]})

    assert response.status_code == 404


def test_install_route_returns_409_when_session_is_busy(tmp_path: Path) -> None:
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
    app.extensions["quanta"]["runs"].run_manual = lambda *_args, **_kwargs: False

    response = app.test_client().post(f"/api/sessions/{session['id']}/packages", json={"specs": ["seaborn"]})

    assert response.status_code == 409


def test_install_route_starts_task_and_records_installed_package(tmp_path: Path) -> None:
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
    captured = {}

    def fake_run_manual(session_id, task, message="Executing code"):
        captured["session_id"] = session_id
        captured["message"] = message
        task()
        return True

    app.extensions["quanta"]["runs"].run_manual = fake_run_manual
    app.extensions["quanta"]["packages"].install = lambda specs, progress: {
        "status": "ok",
        "packages": [{"name": "seaborn", "version": "0.13.2"}],
        "log": "Installed 1 package",
    }

    response = app.test_client().post(f"/api/sessions/{session['id']}/packages", json={"specs": ["seaborn"]})

    assert response.status_code == 202
    assert response.get_json() == {"started": True}
    assert captured == {"session_id": session["id"], "message": "Installing packages"}
    recorded = store.list_packages()
    assert len(recorded) == 1
    assert recorded[0]["name"] == "seaborn"
    assert recorded[0]["version"] == "0.13.2"
    assert recorded[0]["session_id"] == session["id"]
    events = [record["type"] for record in store.events_after(session["id"], 0)]
    assert "package_install_started" in events
    assert "package_install_finished" in events


def test_uninstall_route_returns_409_when_session_is_busy(tmp_path: Path) -> None:
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
    app.extensions["quanta"]["runs"].run_manual = lambda *_args, **_kwargs: False

    response = app.test_client().delete(f"/api/sessions/{session['id']}/packages/seaborn")

    assert response.status_code == 409
