from quanta.app import create_app


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=5000, threaded=True)


__all__ = ["create_app", "main"]
