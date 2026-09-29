"""The web UI's server: Research and a result's popup from the app's database, the re-runs started and stopped from
the page, and every change pushed to the open pages.

    python -m server                      # http://127.0.0.1:8600 (make serve)

Reads the database with query-only connections; writes only a run's row (start, stop, resume). One process: the
background watcher (server.live) and the kept rows (server.research) live in it.
"""
from __future__ import annotations

import asyncio
import gzip
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles

from strategy_lab import db, log
from strategy_lab.config import ROOT_DIR

from server import schemas
from server.live import Live, judge_command
from server.research import Research
from server.runner import NotFound, Refused, Runner, rerun_command

LOG = log.get("server")
DIST = ROOT_DIR / "ui" / "web" / "dist"
PORTFOLIOS = ROOT_DIR / "ui" / "mockup" / "portfolios.html"
RETRY_MS = 2000                 # how soon a page reconnects to a server that went away
CHART_LINES = 16                # the most lines Research's P&L chart draws: its eight colours, solid and dashed


def create_app(db_path: Path | None = None, *, run_command: Callable[[int], list[str]] = rerun_command,
               judge: list[str] | None = None, watch: bool = True, dist: Path = DIST) -> FastAPI:
    path = Path(db_path or db.DB_PATH)
    research = Research(path)
    runner = Runner(path, run_command)
    live = Live(research, runner, judge=judge if judge is not None else judge_command())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db.connect(path).close()                   # the file exists and has the current schema before anyone reads
        task = asyncio.create_task(live.run()) if watch else None
        if watch:                                  # the first page opened gets its rows kept already
            asyncio.get_running_loop().run_in_executor(None, _warm, research)
        try:
            yield
        finally:
            live.close()
            if task is not None:
                task.cancel()

    app = FastAPI(title="Strategy Lab", version="1", lifespan=lifespan)
    app.state.research, app.state.runner, app.state.live = research, runner, live
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

    # ------------------------------------------------------------------------------------------------ research
    def rows(view: str, request: Request) -> Response:
        etag, body = research.rows_payload(view)
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        headers = {"ETag": etag, "Cache-Control": "no-cache", "Vary": "Accept-Encoding"}
        if "gzip" in request.headers.get("accept-encoding", ""):
            return Response(body, media_type="application/json", headers=headers | {"Content-Encoding": "gzip"})
        return Response(gzip.decompress(body), media_type="application/json", headers=headers)

    @app.get("/api/research/meta", response_model=schemas.Meta)
    def meta() -> dict:
        return research.meta()

    @app.get("/api/research/lists", response_class=Response, responses={200: {"model": schemas.ListRows}})
    def list_rows(request: Request) -> Response:
        """Every result of a strategy on one of our lists, judged, in the board's order."""
        return rows("lists", request)

    @app.get("/api/research/assets", response_class=Response, responses={200: {"model": schemas.AssetRows}})
    def asset_rows(request: Request) -> Response:
        """Every strategy alone on an instrument, once a (strategy, instrument, timeframe)."""
        return rows("assets", request)

    @app.get("/api/research/result", response_model=schemas.ResultView)
    def result(key: str) -> dict:
        got = research.result(key)
        if got is None:
            raise HTTPException(404, f"no result {key}")
        return got

    @app.get("/api/research/curves", response_model=schemas.Curves)
    def curves(keys: Annotated[list[str], Query(max_length=CHART_LINES)]) -> dict:
        return research.curves(keys)

    # ------------------------------------------------------------------------------------------------ runs
    def refused(e: Refused) -> JSONResponse:
        return JSONResponse({"message": e.message, "confirm": e.confirm}, status_code=409)

    @app.get("/api/runs/current", response_model=schemas.RunCurrent)
    def run_current() -> dict:
        conn = db.connect(path, readonly=True)
        try:
            return {"run": runner.snapshot(conn, live.code_sha or None)}
        finally:
            conn.close()

    @app.post("/api/runs", status_code=202, response_model=schemas.RunAccepted,
              responses={409: {"model": schemas.Refusal}})
    def run_start(body: schemas.RunStart):
        try:
            rid = runner.start(single_assets=body.single_assets, everything=body.everything,
                               only=body.only.model_dump(exclude_none=True) if body.only else None,
                               confirm=body.confirm)
        except Refused as e:
            return refused(e)
        return {"id": rid}

    @app.post("/api/runs/{run_id}/stop", status_code=202, response_model=schemas.RunAccepted,
              responses={404: {}, 409: {"model": schemas.Refusal}})
    def run_stop(run_id: int):
        try:
            runner.stop(run_id)
        except NotFound as e:
            raise HTTPException(404, str(e)) from e
        except Refused as e:
            return refused(e)
        return {"id": run_id}

    @app.post("/api/runs/{run_id}/resume", status_code=202, response_model=schemas.RunAccepted,
              responses={404: {}, 409: {"model": schemas.Refusal}})
    def run_resume(run_id: int, body: schemas.RunResume):
        try:
            runner.resume(run_id, confirm=body.confirm)
        except NotFound as e:
            raise HTTPException(404, str(e)) from e
        except Refused as e:
            return refused(e)
        return {"id": run_id}

    # ------------------------------------------------------------------------------------------------ live
    @app.get("/api/events", response_class=EventSourceResponse)
    async def events() -> AsyncIterator[ServerSentEvent]:
        """hello on connecting; then results, judged, code, run and reset as they happen (server.live)."""
        q = live.subscribe()
        try:
            yield ServerSentEvent(event="hello", data=live.hello(), retry=RETRY_MS)
            while True:
                ev = await q.get()
                if ev is None:
                    return
                yield ServerSentEvent(event=ev.name, data=ev.data)
        finally:
            live.unsubscribe(q)

    # ------------------------------------------------------------------------------------------------ the page
    @app.get("/portfolios", include_in_schema=False)
    def portfolios():                           # a mockup: a page of its own, with no data behind it
        return FileResponse(PORTFOLIOS)

    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/", include_in_schema=False)
        def no_page():
            return HTMLResponse("<p style='font:14px system-ui;margin:40px'>The page is not built: <code>make ui"
                                "</code>, then reload.</p>", status_code=503)
    return app


def _warm(research: Research) -> None:
    try:
        research.rows_payload("lists")
        research.rows_payload("assets")
    except Exception:                          # noqa: BLE001 - a request will work them out and show the error
        LOG.exception("could not work out the rows ahead of the first page")
