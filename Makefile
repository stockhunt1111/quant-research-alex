PY := .venv/bin/python

.PHONY: setup test lint seed board benchmarks serve ui api rerun service service-restart service-stop

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[ml,dev,web]"

test:
	$(PY) -m pytest

lint:              ## ruff over the Python; the page's React rules (oxlint) and its types
	.venv/bin/ruff check .
	cd ui/web && npm run --silent lint && npx tsc --noEmit

seed:              ## build the bar store from the previous project's disk (no network)
	$(PY) -m strategy_lab.data.seed

board:             ## print the research summary from the database: every result on the research lists, ranked
	$(PY) -m strategy_lab board

benchmarks:        ## three simple reference rules on the widest list of four markets, on 1h, 4h and 1d
	$(PY) -m strategy_lab run sma_cross ibs_reversion donchian_breakout -u etf_core fx_majors crypto_top100 us_stocks_top100 -t 1h 4h 1d

serve:             ## the web UI on http://127.0.0.1:8600 in this terminal: Research live from db/app.sqlite, re-runs there
	$(PY) -m server

service:           ## the web UI as this Mac's service: started at login and whenever it stops, log in logs/server.log
	$(PY) -m server.service install

service-restart:   ## the service on the code on disk again (after a change of server/ or strategy_lab/; make ui needs none)
	$(PY) -m server.service restart

service-stop:      ## the service stopped, and no longer started at login
	$(PY) -m server.service stop

ui:                ## build the web UI's page (ui/web/dist) that the server serves
	cd ui/web && npm ci && npm run build

api:               ## after a change of the server's answers: its description (ui/web/openapi.json) and the page's types
	$(PY) -c "import json; from pathlib import Path; from server.app import create_app; print(json.dumps(create_app(Path(':memory:'), watch=False).openapi(), indent=1))" > ui/web/openapi.json
	cd ui/web && npm run api

rerun:             ## every strategy on every list (SINGLE=1: each on each asset alone too; EVERYTHING=1; RESUME=<run id>; LISTS=, NAMES=, TFS=: those only)
	$(PY) scripts/rerun.py $(if $(RESUME),--run-id $(RESUME),$(if $(SINGLE),--single-assets) $(if $(EVERYTHING),--everything) $(if $(LISTS),-u $(LISTS)) $(if $(NAMES),--names $(NAMES)) $(if $(TFS),-t $(TFS)))
