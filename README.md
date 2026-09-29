# strategy-lab

A strategy in a few lines, measured against the firm's target in one call.

```python
from strategy_lab import indicators as ind
from strategy_lab.strategy import rule
from strategy_lab.evaluate import evaluate

@rule(grid={"fast": [10, 20, 50], "slow": [100, 200], "stop": [None, 0.05]})
def sma_cross(bars, fast, slow):
    """Long while the fast SMA is above the slow one."""
    return (ind.sma(bars.close, fast) > ind.sma(bars.close, slow)).astype(float)

ev = evaluate(sma_cross, "etf_core", "1d")
print(ev.summary())
```

## Setup

```bash
make setup      # Python 3.12 venv with pinned dependencies (+ lightgbm, pytest, ruff, the web server)
make seed       # build the bar store from the previous project's disk (no network)
make test       # offline test suite
make ui         # build the web page (Node; once, and after a change of ui/web)
make serve      # http://127.0.0.1:8600: Research, a result's popup and the re-run, live from the database
make service    # the same as a service of this Mac: started at login and whenever it stops (log: logs/server.log)
```

`make service-restart` puts the service on the code on disk after a change of server/ or strategy_lab/ (a page built
again needs none), `make service-stop` stops it and keeps it from starting at login.

Every evaluation is saved to the app's database, db/app.sqlite (SQLite in WAL mode; strategy_lab/db_schema.sql says
what each table holds): a result with its figures, daily series, walk-forward windows, trades and robustness
measurements in one transaction, so a reader never sees half of one. What is judged from them (a check passed, the
deflated Sharpe, whether a result is stale) is worked out when it is read, so a threshold can change without a re-run.
Nothing a run makes is kept anywhere else: the pages read the database.

Is the data whole: `python -m strategy_lab.data.check` (read-only, no network, about two minutes; `--out FILE.md`
keeps the report). For every agreed list and research timeframe it counts, on the days the list holds each name:
names held with no bars of that timeframe (a ranked list is ranked from what is stored, so another name would take
the seat unseen; who the list holds is read from its daily ranking), series shorter than the vendor's history, holes
against the market's calendar, series that stopped although the instrument still trades, dead markets (an unchanged
price with no volume for three days or more), and in the crypto lists names that are not crypto (a stablecoin, a
token of gold or silver, a contract Binance does not class as a coin). Its exit status is 1 when a list has a
problem.

Fresh data: `python -m strategy_lab.data.refresh binance` (public endpoints) and
`python -m strategy_lab.data.refresh twelvedata` (needs `TWELVEDATA_API_KEY`, `TWELVEDATA_REQUESTS_PER_MINUTE`
in `.env`). Always `--dry-run` first: it prints how many requests the refresh needs and makes none.
Other data commands:
* `refresh binance-gaps` — fills interior gaps of Binance series from the exchange; a gap the exchange has no
  bars for is remembered and not asked again;
* `refresh twelvedata --refetch SYM:TF ...` — re-downloads whole series; a shorter history never replaces a
  longer stored one;
* `refresh twelvedata-seed --timeframes 1h 1d --symbols XAU/USD BRK.B ...` — whole history of series the store does
  not have yet (`--dry-run` gives a ceiling on the requests);
* `refresh twelvedata-extend --timeframes 1h 1d --universes us_stocks_top100 etf_core ...` — history before the first
  stored bar, back to the vendor's first one; the stored first bar is fetched again as a seam (a split since then
  re-fetches the whole series), the earlier part is taken only back to its last hole longer than 10 days, and a
  series the vendor has nothing earlier for is not asked again;
* `refresh binance-seed --markets spot --symbols HYPEUSDT` — a coin new to the store, from the pair's listing, on
  1h, 4h and 1d unless `--timeframes` narrows it (a perp also gets its funding, and ends at its delisting);
* `refresh binance-extend` — a perp series' bars before its first stored one, from the exchange's klines: its monthly
  archives begin in 2020-01, while BTC, ETH and BCH perps trade from 2019; their earlier funding follows;
* `refresh binance-archive --markets spot --symbols FTTUSDT --timeframes 1h` — the missing series of a pair the
  exchange no longer serves (its klines answer "Invalid symbol"), from its public monthly archives; a perp's end at
  its last bar with trades (run `binance-delistings` after it);
* `refresh dukascopy-extend` — the FX majors' hourly history before the vendor's (2020-01), from Dukascopy's hourly
  bid candles, joined only where their closes match the vendor's to 3 bp at the median, the spans it stamped on New
  York's clock put on UTC (`refresh.DUKASCOPY_NEW_YORK_CLOCK`) and its first months, stamped an hour late, an hour
  earlier (`refresh.DUKASCOPY_HOUR_LATE`); 4h and daily rebuilt from it (`--symbols XAU/USD
  XAG/USD XPT/USD XPD/USD WTI/USD`: the commodities' hourly and 4h, their daily bars the vendor's; `--rebuild`: the
  part before the vendor's first hourly bar built anew from the months on disk);
* `refresh cme-hourly --dry-run` — the CME futures list's contracts from Databento (`DATABENTO_API_KEY`): per
  product the most traded contract and the next one, hourly since 2010-06-06, joined across rolls and stored as
  `cme:<product>` with 4h and daily bars built from them, a bar's dollar volume its contracts' notional (contracts x
  the contract's size x the price, `refresh.CME_CONTRACT_SIZE`); priced first, bought only within `--max-usd`, a
  series on disk never bought twice (Databento takes 2-12 minutes to start each: they are asked at once);
* `refresh dukascopy-fill` — the hours the FX majors' and metals' hourly series lack since the vendor's first hourly
  bar (the last hours of Fridays in 2021-07..09, the week's first hours on Sundays, Christmas), from the mid of
  Dukascopy's bid and ask candles where it has one, for the instruments whose Dukascopy prices matched the vendor's
  (not platinum, palladium or crude: Dukascopy's are CFDs on the futures, tens of bp from the spot); 4h and daily
  rebuilt;
* `refresh forexite-fill` — the hours the FX majors' hourly series still lack, from Forexite's free one-minute bars
  (a file a day, kept in data/raw/forexite), only for a pair whose Forexite closes match the stored ones to 3 bp at
  the median; 4h and daily rebuilt;
* `refresh sharadar-tables` — Sharadar's whole sp500, tickers and actions tables (three requests, `SHARADAR_API_KEY`):
  the S&P 500's membership the stock lists use, every security with its permaticker, the corporate actions;
* `refresh sharadar-stocks` — Sharadar's daily bars of every S&P 500 member since 1996, those that no longer trade
  included (one request a ticker, kept in data/raw/sharadar/stocks so a stopped run resumes), stored as
  `sh:<ticker>` with their dividends and spin-off values; a member's Twelve Data hourly bars are put on that daily
  basis and stored as its `sh` 1h and 4h;
* `refresh etf-daily --dry-run` — the 34 ETFs' daily bars from three sources, each kept on disk: Sharadar's funds
  table (data/raw/sharadar/funds), Twelve Data's (data/raw/twelvedata/daily) and each venue's own daily bars since
  2018-05-01 (Databento, data/raw/databento/daily; `--max-usd`), decided day by day (`refresh.etf_daily_reconciled`)
  and stored as their `td` daily bars, their hourly bars then put on them; the Twelve Data refresh leaves these daily
  series alone;
* `refresh databento-hourly --dry-run` — the hourly sessions the lists' US listings lack since 2019-01-07, from
  Databento's one-minute bars of seven lit venues (`DATABENTO_API_KEY`): a member that left the market, the years
  before a ticker change Twelve Data serves only since, the years before a ticker was taken up by another company,
  the ETFs' year before Twelve Data's first hourly bar, the days Twelve Data skips. The dry run prices them; a real
  run needs `--max-usd` and keeps every answer in data/raw/databento/minutes, so nothing is bought twice;
* `refresh binance-delistings` — the exchange's contract list (one request): a perp Binance lists as SETTLING or
  CLOSE ends at its deliveryDate, one no longer in the list at its last trade; after that Binance goes on printing
  its last price with no volume, which is not a market. The date goes into the series' meta, where a backtest
  closes the position and a list gives up the seat;
* `refresh dividends` — the cash dividends of every stock and ETF the agreed lists ever hold: Yahoo Finance's amounts
  (the whole history in one request), except spin-offs the stored prices already carry as a split
  (`refresh.CARRIED_BY_PRICES`); a dividend on any other Twelve Data split date is paid and flagged for that review;
  one that never paid is stored empty;
* `refresh twelvedata-first-bars` — the vendor's first hourly bar of every Twelve Data series of the agreed lists
  (data/reference/twelvedata_first_bars.csv, one request a symbol not yet in it): the data check's measure of a
  short hourly history;
* `python -m strategy_lab.data.minute_refresh binance|forexite|twelvedata --dry-run` — the one-minute bars the engine
  walks intrabar exits through (`data.minutes`: memory-mapped arrays in data/store/<source>/1m, one copy for every
  process of a batch): the crypto lists' perps for the months each is held and the month after (Binance's public
  archive, a file a month in data/raw/binance/minutes), the FX majors since 2003 (Forexite's free files,
  data/raw/forexite), the stocks, ETFs and spot metals of the lists since 2020-03-24, Twelve Data's first minutes
  (about 25,600 requests, twelve at once within `TWELVEDATA_REQUESTS_PER_MINUTE`; a month a file in
  data/raw/twelvedata/minutes; the vendor refuses the listings that no longer trade: ATVI, PXD, TWTR, WBA, XLNX).
  Every hour's minutes are held to the stored hour: Binance's minutes with a trade make its open, high, low and close
  to single precision (a minute without one repeats the last close and is left out: DOGE traded in 27% of its spot
  minutes of 2019-09, and its hours open at their first trade);
  Twelve Data's are first put on the stored hours' basis session by session (a stock's stored hours are the vendor's
  rescaled where its intraday series misses a corporate action, or another source's), then make its high and low (the
  stored hour opens at its session's auction, a minute's first trade is not the auction); an hour they do not make,
  and an hour inside their span with none, is kept as one bar of the hour's own prices, and minutes of an hour the
  store does not have are left out. Forexite's, another feed, are kept month by month where their hours line up with
  the stored clock and their closes keep within 3 bp of one ratio to the stored ones at the median, and are put on the
  stored side of the quote by that ratio (its AUD/USD stands 2-4 bp above Dukascopy's bid through 2003-2005);
* `python -m strategy_lab.data.spread_refresh exness|dukascopy|build --dry-run` — the spread a broker quoted on the
  five spot quotes of commodities, hour by hour, which a fill on them pays (`data.spreads`, `engine.costs`): Exness's
  ticks (its Pro account: no commission, so the spread is the whole cost; a zip a quote and month in data/raw/exness,
  free and with no key: 618 months from 2015-08 for gold and silver, 2016-01 for platinum and palladium, 2019-02 for
  crude, 26 of them not in its archive, 5.2 GB), Dukascopy's hourly bid and ask candles before them (gold from 2003-05, silver from 2011, crude from
  2013-10, data/raw/dukascopy), put on Exness's level by the ratio of the two brokers' spreads over Exness's first
  year; `build` writes each quote's hours (the spread at the first and the last quote of each, in dollars; an hour
  without an ask above the bid is left out) to data/store/td/spread. The same ticks give crude's prices
  (`refresh.EXNESS_QUOTES`): their mid's minutes, built once a month and kept in data/raw/exness/<symbol>/minutes,
  make its hours whenever its hourly series is written (`refresh twelvedata`) and its minutes (`minute_refresh
  twelvedata`);
* `python -m strategy_lab.data.market_cap stocks|coins` — today's market capitalisations (Nasdaq's stock screener,
  CoinGecko; one request each, no key) behind `us_stocks_mcapN` and `crypto_mcapN`;
* `python -m strategy_lab.data.rates` — the 3-month T-bill rate (FRED DTB3; one request, no key) that idle cash earns
  when a record is compared with buy-and-hold at equal risk;
* `python -m strategy_lab.data.orderly markets|compare` — CoinIQ's market list (Orderly) and a per-coin check
  that the Binance perp prints the same prices.

## Writing a strategy

* `@rule` — the function gets one instrument's bars (`open high low close volume dollar_volume`, indexed by bar
  close time) and returns a position in [-1, 1] per bar, decided at that bar's close; it must not write into those
  bars, which the grid's signal configurations share (tests/test_lookahead.py checks every rule). An indicator of
  `strategy_lab.indicators` or a regime of `strategy_lab.regimes` asked for on the bars or their columns is worked out
  once for all of them (`strategy_lab.shared`; on a series derived from the bars, once per call). Across a universe each
  instrument gets an equal share of capital among the instruments in the universe at that bar. Grid key `slots` = k
  gives the book k slots of 1/k of capital instead: a trade takes a free slot on its first bar and keeps it, at that
  share, until it ends; a trade that finds every slot taken is not traded; of the trades starting on one bar the
  more liquid instrument's takes a slot first (its median daily dollar volume over 60 days, a currency pair's
  market turnover from the BIS survey). A list runs only the slot counts below its number of names (5 slots from
  six names, 10 from eleven, 20 from twenty-one; a Top-3 list and the five spot commodities one slot per name only):
  as many slots as names or more would not put the capital on fewer names, only leave part of it idle. For the same
  reason a strategy that holds its top `top_k` names (`dual_momentum`, `sector_rotation`) holds at most as many as the
  list has.
* `grid` values — three for each numeric parameter, the source's standard in the middle (Donchian 20/55/100 around the
  Turtles' 55, Keltner k 1.5/2.0/2.5, ADX 20/25/30 around Wilder's 25, a chandelier trail 2/3/4 ATRs around LeBeau's
  3), five for a strategy's only one (RSI(2) entry 2.5-20, a Bollinger k 1.5-2.5, the one-bar IBS threshold 0.1-0.3,
  the model's threshold 0.5-0.7, TSMOM's 3-24 months around the source's 12, Faber's 6-14 around his 10), two for a
  switch (long only or not); a stop is none, 2 or 3 ATRs (the Turtles' 2N). Two values a parameter had been too few
  (user, 2026-09-27). No rule keeps its source's values alone: `ibs` with its two graded forms (0.15-0.25 and
  0.75-0.85 around the source's fifths) and `gtaa_faber` had, to be compared with the research desk's IBS and Faber's
  paper as they are, and are searched since 2026-09-28 (user: no rule is compared with a reference any more, our data
  and markets differing from theirs). Exits added to a rule that has none earned no place in its grid (2026-09-27,
  walk-forward on the 15 lists x timeframes of one list per market): a chandelier trail of 3 ATRs lowered the
  out-of-sample Sharpe of `sma_cross`, `ema_trend`, `calm_trend`, `late_entry_trend`, `keltner_breakout` and
  `donchian_breakout` by 0.06-0.39 at the median (better on 1-6 of 15), a target of 3 ATRs by 0.01-0.34
  (`keltner_breakout` +0.01, 8 of 15); a stop of 2 ATRs lowered `rsi2_connors`, `bollinger_reversion`, `ibs`
  and `trend_or_revert` by 0.12-0.47 (better on 1-2 of 15; Connors found stops hurting), a target of 2 ATRs by
  0.01-0.23. A trend rule earns from the few long moves a trail or a target cuts short; a mean reversion's stop sells
  the dip it bought. Nor did a trailing stop switched on once a trade is in profit (user, 2026-09-27: on at +2, 4 or
  8% trailing half that, or on at 1-4 ATRs trailing 0.5-2), set as well as a trend rule's own exit or riding past a
  mean reversion's: on the same 15 lists x timeframes it made the trend rules' drawdowns shallower by up to 33 pp at
  the median but cost them up to 0.41 pp a month of return and up to 0.52 of Sharpe, and changed the mean reversions'
  little; the one variant better on all three there (`donchian_breakout` on at 4 ATRs trailing 2: +0.10 pp a month,
  a drawdown 13.6 pp shallower, Sharpe +0.05) did worse on nine other lists (on 4h and 1d, its level moving each
  hour: -0.16 pp a month and -0.12 Sharpe at the median, better on 4 and 5 of 18; on 1h: Sharpe -0.13, the same
  return); walked through minutes on crypto Top-10 1h, every variant of `sma_cross`, `ema_trend`, `late_entry_trend`,
  `trend_or_revert` and `tsmom` lost return and Sharpe whatever the order inside a minute.
* Time — a rule's indicator periods are bars of the timeframe it runs on, as on any chart.
  A span a strategy's source gives in days or months keeps its length in time on every
  timeframe: `bars_in(bars, days=200)` or `bars_in(p, months=12)` is the number of bars it takes on the market and
  timeframe run (`config.BARS_PER_DAY`: a US session 7 hourly bars and 2 four-hour ones, a day of crypto or FX 24 and
  6, of commodities 23 and 6; a month is a twelfth of `config.TRADING_DAYS`, 21 sessions of a stock, 30.4 days of a
  coin): Connors' 200-day average, the 12- and 6-month returns of `dual_momentum`, `sector_rotation` and `tsmom`, the
  months of `xs_momentum`, `betting_against_beta` and `vol_managed`. A decision the source takes once a month or week
  is taken at the period's first close (`rebalanced(w, "M")`, `period_starts`): Faber's, Antonacci's, TSMOM's and
  Moreira and Muir's monthly signals, the panels' rebalances. Until 2026-09-27 every span was a count of bars: on 1h
  GTAA's 10-month average was 210 hours, a "monthly" rebalance came every 21 hours, and Connors' trend 8 days of a
  coin.
* `long_only` — a rule with a mirror trades its short side too unless the switch says long only (`with_short`: the
  long side minus the short one), and the walk-forward chooses on each list's past: `sma_cross` and `breakout_trail`
  short below their average or channel, `tsmom` short on a negative trailing return as its source, `ibs_reversion`
  short on strength, the direction models short while the probability of a rise is below 1 - their threshold
  (`ml_direction`, and `ml_feature_search`, whose search tries long only and long/short, trading
  the model's "not up" as a short). A rule's short side is its long side on the prices turned upside down
  (tested). Nine rules had the switch from 2026-09-27 and trade long only since 2026-09-28 (user: remove it where it
  does worse): `ema_trend`, `calm_trend`, `late_entry_trend`, `donchian_breakout`, `keltner_breakout`,
  `big_move_follow`, `rsi2_connors` (Connors sells RSI(2) above 95 under the 200-day average), `bollinger_reversion`
  and `trend_or_revert`. Measured walk-forward on the 15 lists x timeframes of one list per market (FX majors, CME
  futures, crypto Top-10, stocks Top-10, the 34 ETFs; `big_move_follow` on its own three lists), the rule as it was
  against the same rule held to its long side: in the 82 of their 129 cells where the walk-forward took the short
  side, the long side alone made more a month in 67 (less in 10; +0.07 pp at the median), drew down less in 53
  (deeper in 17; 3.0 pp), had the higher Sharpe in 67 (lower in 12; +0.07), and on average as well; one short side
  lost more than the account (`big_move_follow` on crypto Top-100 1d, a coin's surge, before a short was liquidated). `sma_cross`, `tsmom` and
  `breakout_trail` did the same in most of their cells (27 of 35 a month, Sharpe 28 of 35) but not on average: their
  short side paid on crypto, `tsmom` 1.2-1.3 pp a month on crypto Top-10 on every timeframe, the walk-forward short in
  every window there; `ibs_reversion` took its short side in 2 of 15 cells. The allocation strategies
  (`dual_momentum`, `sector_rotation`, `gtaa_faber`, `vol_managed`), `ibs` (held to its exit as `rsi2_connors` and
  `bollinger_reversion`, whose short side did worse) and the rules graded by a model stay long only. The short side pays
  what the engine charges a short (borrow on stocks and spot coins, funding on perps, the dividends); a currency
  pair's interest carry is not modelled either way. Only long-only had been written until 2026-09-27, carried over
  from the earlier project, where long-only beat long-short for trend rules on crypto and equities, markets that drift
  up (user, 2026-09-27: strange that 22 strategies traded long only).
* `@panel` — the function gets the whole universe (`p.close` etc. are wide frames) and returns weights with
  sum |w| <= 1. Add a `live` argument to receive the universe membership mask. The weights are a book rebalanced
  whole (every position traded back to its weight when any weight moves: an allocation decided once a month or week),
  or with `@panel(book=False)` positions of their own, each traded only when its own weight moves (`big_move_follow`'s
  events: rebalanced whole, a book of events sold its running moves down at every new one, 22-47% of the bars it held
  anything on 1d lists, 2.7% a year on crypto Top-50 at the median).
* `grid` — every combination is evaluated; walk-forward picks among them. Keys `stop`, `take`, `trail` (fractions of
  the entry price) and `stop_atr`, `take_atr`, `trail_atr` (multiples of the instrument's 14-bar average true range:
  the same room relative to its swings on an hour of a currency pair as on a day of a coin) switch on the engine's
  intrabar exits, and `trail_every` (`bar` or `minute`) says when a trailing stop's level is re-set (Engine
  conventions). A rule's trade ends at its exit before the capital is shared out: a trade stopped out gives its seat
  or slot back at once, and waits for its rule to turn (flat or to the other side) before it trades again, whatever
  happens to its share of the capital. The exit is the rule's trade's, measured from the trade's own entry, and the
  engine closes the position there at that price, a name seated in the middle of its rule's trade included.
* `@rule(exposure=True)` — the rule's position is a holding kept while the name is listed, only resized, never an
  entry or an exit (`vol_managed`): on a list that changes its names are the list's members, a leaver sold at the
  re-pick and a newcomer bought at once, instead of a leaver keeping its seat until its trade ends.
* `@rule(grade=...)` — a model grades the rule's own trades before the universe's seats are given out
  (`strategy_lab.trade_model`): `take` keeps the trades it gives a probability above the grid key `threshold` of
  making money after costs, `size` scales each by 2p - 1. One model learns from every instrument of the run at once
  (a list's names pooled, an instrument alone from its own trades), refit every 3 months on the trades already closed.
  `ibs_ml_filter`, `ibs_ml_sized`, `rsi2_ml_filter`, `rsi2_ml_sized`, `bollinger_ml_filter`, `bollinger_ml_sized`,
  `trend_or_revert_ml_filter`, `trend_or_revert_ml_sized` are IBS, RSI(2), the Bollinger reversion and the regime
  switch graded so, each its rule written again (long only, as the four rules are): rules that trade often and win
  more trades than they lose. A trend rule wins a third of its trades and earns on its few large ones: a model that
  takes a trade only above one half would leave most of them out.
* A bar an instrument missed (vendor gap, halt) is not a decision point for it: a rule keeps its last position,
  a panel strategy sees the instrument's last close.
* Indicators: `strategy_lab.indicators` wraps TA-Lib (150+ functions: `ind.call("CCI", h, l, c, timeperiod=20)`).
* One strategy per file: `strategies/<name>.py` holds the strategy `<name>` alone, so that it can be read, or handed
  to another engine's developer, on its own; the building blocks several strategies share (`hold_between`,
  `with_short`, `rebalanced`, `long_short`, `bars_in`) are in `strategy_lab.strategy`. Commands and scripts name a
  strategy by its name.
* Every strategy in `strategies/` is checked by `make test` for look-ahead: its positions must not change when the
  future is cut off.

Run from the command line: `python -m strategy_lab run ema_trend tsmom -u etf_core crypto_top10 -t 1d 4h`.
Universes: `etf_core`, `etf_topN` (the N most liquid of etf_core each month), `fx_majors`, `fx_all`, `us_stocks_topN`
(S&P 500 members, top N by liquidity each month),
`crypto_topN` (USD-M perps, top N by liquidity each month; not stablecoins, tokens of gold or TradFi contracts), `us_stocks_mcapN` (today's N largest S&P 500 members by
market capitalisation, one share class per company), `crypto_mcapN` (today's N largest coins by market capitalisation
with a Binance USD-M perp, stablecoins skipped: traded on perps as every crypto list, bought and held on spot),
`coiniq` (CoinIQ's coins, priced with the Binance perp of the same
coin), the markets the firm's research desk (Stockhunt) scores strategies on, with its own instrument
lists — `stockhunt_stocks` (its 216 US names, 100 held at a time by liquidity: the desk does not publish its own
rule), `stockhunt_etfs` (SPY QQQ IWM TLT XLF GLD EFA DIA XLV XLE), `stockhunt_crypto` (its 20 coins, on Binance
spot), `stockhunt_commodities` (gold, silver, platinum, palladium, WTI crude, spot quotes, the prices the firm's
simulator fills at) — our `cme_futures` (the CME futures of gold, silver, platinum, palladium, WTI crude and copper,
their contracts joined; the desk's own CME market holds 19, five of them equity index futures) — or a single id like
`td:SPY`, `perp:BTCUSDT`, `cme:GC`.

The research runs on the lists of `strategy_lab.lists`, by market: stocks and crypto Top-3, Top-10, Top-50 and
Top-100 by liquidity (crypto's Top-3: BTC and ETH, which have not left it since 2021, and the third most traded coin),
ETFs' Top-3 of the 34 by the same rule, re-picked monthly with a buffer on the daily bars whatever the timeframe, so a
list holds the same names on 1h and 4h as on 1d (a session's intraday bars miss its auctions, a share of its volume
that differs by name) (`us_stocks_top3/10/50/100`, `crypto_top3/10/50/100`, `etf_top3`: a member keeps its seat
until it ranks below 1.5 x N or stops trading, so a name at the edge no longer flips in and out every month; a seat
goes only to a name that traded in the window, and a delisted market gives up its seat the day after its last bar to
the best-ranked name outside); ETFs also as the 10 majors
(`stockhunt_etfs`) and all 34 (`etf_core`); the 7 FX majors; all 5 commodities on their spot quotes
(`stockhunt_commodities`) and 6 on their CME futures (`cme_futures`), two markets as on the desk. Liquidity,
not market capitalisation: a list that changes through history needs its ranking at every past date, and
capitalisation is known here only for today — today's largest names carried back in time are the winners picked with
hindsight. The desk's own stocks and coins (`stockhunt_stocks`, `stockhunt_crypto`) are not research lists: they run
only when asked for. Nothing runs on another list unless the user asks for it: the command line and the batch scripts refuse a list outside these (for each instrument alone:
the widest list of each market and the ML task's markets, `lists.PER_INSTRUMENT`) without `--outside-lists`.

## Each strategy on each instrument alone

The firm's products, dashboard and simulator run one strategy on one symbol with its whole capital. The same test:
`per_asset.evaluate(strategy, universe, timeframe)` scores a rule strategy on every instrument of the universe
alone (parameters by walk-forward on that instrument's own past, out-of-sample only, against the target and the
instrument's own buy-and-hold, cash for a currency pair or crude's spot quote; `slots` dropped). A ranked universe contributes
today's members, each over its whole history. `python scripts/per_asset.py` runs every rule strategy on the widest
list of each market
(`strategy_lab.lists.PER_ASSET`: today's 100 most liquid S&P 500 stocks, today's 100 most liquid Binance coins, the 34
ETFs, FX majors, the five commodities' spot quotes and the six CME futures) on 1h/4h/1d — the narrower lists of a
market are inside its widest one; Research shows them with Run on: Assets.

The ML task asks for the best strategy for each of its instruments with as little fitting as possible, each
instrument alone: today's 10 largest US stocks and coins by market capitalisation and the 7 FX majors,
and the five commodities the firm's simulator fills at and its ten ETFs,
`lists.ML_TASK`; it runs in a re-run with single assets.
`strategy_pick` answers it the way a strategy's parameters are chosen, one level up: every strategy run on the
instrument alone (the rules and `ml_feature_search`) is a candidate with its own walk-forward record; every 91 days,
from the day the candidates have a year of records, the one with the best Sharpe on all the days before is held until
the next choice (a candidate with a position on fewer than 60 of them is not chosen), and a switch pays the
instrument's cost on the change of position. The stitched record is what choosing the best strategy would have earned.
`python -m strategy_lab db pick` (the picks stage of a re-run with single assets) saves one record per instrument and
timeframe of the ML task's lists; singling out the best of them is a choice again, judged against the luck of their
own tries (family picks).

`ml_feature_search` measures a per-symbol ML engine, one that picks a model and an indicator combination for each
symbol: LightGBM on every combination of five indicator families, the combination re-picked every 3
months by walk-forward on the symbol's own past. `python scripts/per_asset.py --names ml_feature_search
--universes us_stocks_mcap10 crypto_mcap10 fx_majors stockhunt_commodities stockhunt_etfs` runs it; `python
scripts/product_protocol.py` with the same lists replays a 70/30 protocol on the same matrix (the combination with
the best Sharpe on the last 30% of the history, over the 30% and over the whole history) and writes
reports/product_protocol.md: its figures next to what the walk-forward gets. A pick can also be checked on similar
symbols; `python scripts/peer_check.py` measures that idea as a rule inside the
walk-forward (`walkforward.choose_with_peers`: a configuration must have beaten holding on at least half of the other
names of the list over the window it is chosen on; on a currency pair, whose buy-and-hold is cash, made money) with
and without it, and writes reports/peer_check.md.

## What the numbers mean

* **Out-of-sample (OOS)** — walk-forward with a growing window: parameters are first chosen on the first year of
  history, then re-chosen every 3 months on all the history before each window
  (`config.WF_SCHEMES`). A rolling window (the scheme's `train`: the last year, or three years once the history is
  that long) does no better: on each strategy on one list of each market, the timeframes in turn (99 results,
  2026-09-27), the rolling year's Sharpe was lower on 61 and higher on 38 (median -0.04), the rolling three years'
  lower on 69 and higher on 27 (median -0.03); beyond noise (|t| >= 2) 2 higher and 4 lower, 0 and 3, about what
  luck gives in 99; lower for 14 and 15 of the 19 strategies; its choice changed a median 22 and 13 times where the
  growing window's changed 6 (`python scripts/rolling_walkforward.py` -> reports/rolling_walkforward.md). This is the
  number compared with the target. An instrument is judged once it has at least 6 months out-of-sample. A list is
  scored from the first day it holds at least half of its names (half of a top-N list's N, half of a fixed list): the
  S&P 500 lists from 1996, where the membership data begins, the 10 major ETFs from 1998-12, all 34 from 2002-07, the
  spot commodities from 1983-03 (1h and 4h from 2019-02-18), the CME futures from 2010-06; older bars serve only as
  indicator history, since a basket of one or two names is not the list.
* **In-sample (IS)** — the parameters that are best over the whole history, scored over the out-of-sample days (the
  same days as the OOS record, so the two compare). The gap between IS and OOS is how much of the result was fitting.
* A window before which no configuration held a position on 60 days of the history (a model's first year, before it
  has trades to learn from; a 12-month signal's) has nothing to be chosen on and holds no position; its parameters
  show as none. The record's trades are those of the positions the windows held one after another, as one account
  holds them: a position a switch of configuration keeps on its side runs on as one trade, one it closes ends there.
* **Target** (`config.FIRM_TARGETS`): average month >= 1.5%, >= 70% green months, max drawdown >= -10%,
  Sharpe >= 1, beats buy-and-hold of the same universe: more money than holding it once sized to its risk, the
  research desk's criterion (`metrics.equal_risk`: each day's multiple is buy-and-hold's trailing 90-day volatility
  over the record's, both known the day before, at most 2x; idle equity earns the 3-month T-bill rate, money borrowed
  past 100% pays it + 1.5%), shown as the return a year above holding (`vs_bh`). A strategy in the market half the
  time carries half the risk: its raw return against holding measures how much capital it deploys, not its skill.
  Buy-and-hold (`engine.hold`) puts equal money in the list's instruments when it starts and then holds the units,
  on spot terms: a coin at its Binance spot pair's prices (the perp's own, on the pair's scale, before the pair lists
  or for a futures-only coin), no funding, dividends paid on the ex-date. Money moves only when the composition
  changes, on the bar after the decision: a leaver is sold and its money buys the newcomer; only those trades pay
  costs, at spot rates. It holds no currency pair and no crude's spot quote, which strategies still trade: a pair's
  holder earns the difference of two interest rates (not modelled) while its price goes nowhere over the years, and
  crude is held through futures rolled every month, whose cost its spot price does not show. The CME futures list is
  held whole, crude's future too: its joined contracts carry the rolls, and a future takes margin, not cash, so the
  money test leaves all the equity of both the record and buy-and-hold at T-bills (`hold.margined`), with nothing
  borrowed past 100%. A list with nothing else to hold (FX)
  is compared with cash (`metrics.vs_hold`): the record, not sized (cash has no risk to match), with its idle equity
  at T-bills, against T-bills; its buy-and-hold figures are zero, as the idle cash of every record here. A month
  without a position is
  neutral: a strategy's green share is taken over the months it was in the market (shown next to how
  many months that was); a portfolio of capital is judged on all months. The average month is compounded: the steady
  monthly rate that takes the account from the record's start to its end over the record's length, a partial month at
  either end counted by its days (+100% then -50% is 0%, not the +25% of a plain mean; 1.5% a month is 19.6% a year),
  as the firm's research desk (ROI/yr) and ManifoldBT (CAGR) report returns.
* **The market's index** (`server/market_index.py`) — beside buy & hold in a result's popup (the metrics' third
  column, a dotted line on its chart and on Research's): what a client could have bought instead, bought on the
  record's first day and held over the same days as buy & hold holds one instrument. A list's: SPY with its dividends
  for the stocks and the ETFs, BTC on Binance spot for crypto, gold's spot quote for the spot commodities, gold's joined
  future (GC) for the CME list (a future's return without the interest on its money, as the list's own records are);
  none for FX, which is compared with cash. An asset alone has its own market's, where an index measures that market:
  SPY beside a US stock or an ETF of US stocks, BTC beside a coin; none beside an asset that is its market's index, nor
  beside a metal, a future or an ETF of bonds, commodities, gold miners or foreign stocks, whose own buy & hold is the
  reference (gold's daily returns correlate 0.04 with crude's since 2016, TLT's -0.15 with SPY's: the user asked on
  2026-09-28 why oil was compared with holding gold). WTI's spot quote, which cannot be held, has crude's future held
  across its rolls beside it (CL, 0.89 with the quote): what a holder of crude earned. The two answer different
  questions:
  buy & hold of the same list is what the rule's timing adds to what it trades, and the target is judged on it; the
  index is whether buying the market would have done better, which the list's make-up decides as much as the rule
  (the research desk's top 100 held grew 13.4% a year against SPY's 11.4% over 2003-2026; holding our crypto
  Top-10 from 2021-01-31 lost 15% where BTC gained 149%). The index is shown as it is, not sized to the record's risk:
  a list result keeps its average exposure, not its daily one, and the money test worked out from the average is off
  the exact one by a median 0.23% a year (0.6-0.7% on the stock lists; the sign differed on 21 of 1,032 results,
  2026-09-28). SPY held so comes out at its published total return to within the gap between the close before the
  record's first day and the open it is bought at (2008 -36.96% against -36.79%, 2022 -18.43% against -18.18%; the
  price alone -38.3% and -19.5%).
* **At 10% DD** — the average month over all the out-of-sample years with the record's positions scaled so that its
  max drawdown is the target's (`db.at_target_dd`), to compare returns that come with different drawdowns: Calmar's
  question (a year's return over the worst drawdown, as the research desk, ManifoldBT and the Minerva tester show it)
  in the target's units, which orders the records as Calmar does. A record that fell 25% is held at 40% of its size,
  one that fell 5% at twice it: the multiple is found by halving its span until the scaled daily record draws down
  exactly -10%. The money the positions leave idle earns nothing, as in the record's own average month (a multiple of
  1 gives that month back); money borrowed past the equity pays T-bills + 1.5%, counted on the record's average
  exposure (for an instrument alone saved before it was kept, its share of days in the market, which bounds it). The
  multiple has no ceiling (a ceiling of twice the equity in positions, the money test's, was removed at the user's
  question on 2026-09-28; it had bound 1 result of 15,154): the figure is what the record earns at the target's
  drawdown, and the borrowing is what that leverage costs; a record that never lost a day has none. A row of fewer
  than 30 trades shows none either (`research.MIN_TRADES`, the research desk's bar below which it greys a strategy's
  trade figures, "Under 30 trades none of those is a measurement"; the figure stays kept): its drawdown measures
  nothing, and sized to -10% it read up to +162% a month (`breakout_trail` on HEI 1d, 2 trades, 99 times its size);
  3,337 of the 14,209 single-asset results and 40 of the 945 list results on 2026-09-28. None of the 945
  list results of 2026-09-28 borrows at -10% (at most 85% of the equity in positions on average). The multiple is
  chosen knowing the record's worst drawdown: the figure compares records, it is not a size a forward run could have
  known. A longer record had more years for its worst drawdown to come (at a Sharpe near 1, 28 years draw down about
  1.6 times as deep as 5, by Magdon-Ismail and Atiya's expected maximum drawdown, 2004): on hover the page shows beside
  the figure of all the years the same over the record's last five years alone, the years every research list has out
  of sample (its crypto lists' records begin in 2021). A result saved before the figure gets it from its kept series
  (schema steps 3 and 4). Buy & hold's own figure stands beside it, on the row's hover and in the popup's metrics (the
  index's too): holding, fully invested, scaled the same way, so that the two compare at one drawdown (`rsi2_connors` on
  stocks Top-50 4h +0.90% a month against holding's +0.52%, which reaches -10% with 28% of the money in the list). The
  money test (`vs B&H / yr`) compares at equal volatility instead, sized from the trailing 90 days and at most 2x, as
  the research desk does; the two disagree on 175 of the 997 list results that have both (2026-09-28): 114 beat holding
  at -10% and not in money, 61 the other way. A buy & hold saved before the figure gets it from its kept series (schema
  step 5).
* Returns are fractions of the traded capital with no leverage: a strategy's weights stay within its capital (sum |w|
  <= 1, checked), and no fill takes the positions held past it (Engine conventions). Bar returns are
  compounded into UTC calendar days, so crypto and equities share one annualisation; months are complete calendar
  months only.
* **K-ratio** — Kestner's, as he corrected it in 2013 ("(Re)Introducing the K-Ratio", SSRN 2230949): the
  least-squares slope of the account's log equity day by day, over the slope's standard error, times √365 over the
  number of days (`db.k_ratio`). It measures how steadily the account grew: a record of independent days scores about
  1.1 times its Sharpe (√1.25: a random walk strays from its fitted line by √(n/15) of a day's spread; tested), a
  steadier climb more, one made in a few jumps or broken by long flat or losing stretches less; his 1996 and 2003
  versions grew and shrank with the record's length. Out of sample on the 945 list results (2026-09-28) it ranks them
  much as the Sharpe does (rank correlation 0.91), at a median 0.98 times it where the Sharpe is above 0.3, and parts
  from it where a record earned in bursts: `ibs_ml_sized` on the spot commodities 1h has a Sharpe of 3.31 and a
  K-ratio of 0.64, `tsmom` on stocks Top-3 1h 1.27 and 1.50. The writer works it out from the series it saves, as it
  works out the trades' profit factor (no evaluation reads it, so no result's stamp covers it); the results saved
  before got it from their kept series (schema step 2). None for an account that lost everything, whose log equity
  ends there (4 single-asset `tsmom` shorts on coins).
* **Monte Carlo** — stationary block bootstrap of the OOS daily returns of a result that makes money: P5/P50/P95 of
  the target figures.
* **Makes money** means a positive Sharpe AND a positive compounded return: a positive Sharpe can still lose money
  once large swings compound, and such a result is rejected with that reason.
* **Beyond luck** — every evaluation made is a try the best results were picked from: `P(beyond luck)` is the
  probability that a record's Sharpe is above what the best of that many worthless tries would show over a record of
  the same length (`significance.deflated`: Bailey & Lopez de Prado's deflated Sharpe, with its skew and kurtosis).
  A try that resembles another is not another chance: the same rule on nested lists, on neighbouring timeframes or
  with its model wins and loses with itself. The board counts the independent tries its saved evaluations add up to
  (`board.luck_of`: were none of them skilled, their Sharpe ratios would be normal scores correlated as their daily
  returns over the days both cover, times the share of each record those days are; the count is the number of
  independent scores whose best reaches as high on average: 690 evaluations made 174 tries on 2026-09-25). A strategy
  on an instrument alone is judged against the tries on its instrument (`board.refresh_asset_tries`): every (strategy,
  timeframe) pair ever scored on it, once however many lists scored it, correlated as their records are (72 pairs made
  a median of 31 tries on 2026-09-28). That is the choice its row answers, which strategy for this instrument, as the
  firm's products and the ML task pick one for a symbol. The tries on every instrument at once (pairs of different
  instruments taken as independent though they share their market: 17,035 pairs made 13,234 tries) say how far luck
  takes the best row of the whole table; the probability against them is shown beside the instrument's own.
* **Random timing** — the record's positions against the same positions moved in time (each instrument's path rotated
  within its own bars): same time in the market, same holding periods, only the timing random. The p-value is the
  share of 500 rotations that earn the record's Sharpe (`significance.random_timing`); a record whose rotations do as
  well earns from being in the market, not from its timing.
* **Robustness** — the board's `robustness` column: the checks a result passes of those that apply to it (`7/9`;
  one that does not apply is left out, one whose record is too short to judge counts as not passed). They are
  measured in the evaluation of a result that makes money out-of-sample (`strategy_lab.robustness`, on the
  evaluation's own backtests) and judged when the board is built (`board.robustness`: the thresholds live there, so
  changing one needs no re-run); a result that loses money is not checked (—), nor given Monte Carlo and random
  timing. The checks, and where each comes from (the firm's research desk and engine, the Minerva tester, Build
  Alpha):
  1. Sharpe beyond luck: Monte Carlo 5th-percentile Sharpe above zero, and P(beyond luck) at least 95%;
  2. beats buy & hold beyond luck: the Sharpe above holding the same list over the same days (above zero against
     cash), t (paired block bootstrap, 20-day blocks) at least what luck gives the best of the tries one time in
     twenty: the 95th percentile of their best score were none of them skilled (`board.luck_of`, a familywise test at
     5%, maxT; 3.61 on 2026-09-25);
  3. timing beats random: p below 0.05;
  4. not overfitted: the probability of backtest overfitting (combinatorially symmetric cross-validation of the
     grid's records, 10 blocks) at most 0.5; not for a single configuration;
  5. parameter plateau: the configurations one grid step from the one the windows chose most all make money and keep
     half its Sharpe at the median; not for a single configuration;
  6. stable across eras: the Minerva tester's consistency over two-year windows (a last one of a year counts) at
     least 0.60, three windows needed;
  7. a bar later: still makes money with every position taken a bar later, on the same choices;
  8. costs x3: still makes money at three times the modelled costs (a spot quote of a commodity: three times the
     spreads its broker quoted), a CME future at 10bp a side (thin intraday quotes its flat 2bp understates);
  9. most names make money: at least half of the names it traded made money on their trades;
  10. neighbouring lists: on a Top-N list, Top-(N-d) and Top-(N+d) (d = 20% of N rounded to 5, at least 5 and at
      most half of N: 2/4, 5/15, 40/60, 80/120) make money and keep half its Sharpe; evaluated inside the result's
      evaluation, never saved;
  11. with a model, other seeds: seeds 1-4 of its model make money and keep half its Sharpe;
  12. with a model, beats the plain rule: t at least 2 over the same rule without its model.

  A strategy on an instrument alone (Research's Assets, `per_asset`) is checked the same way, in its own evaluation,
  but for 9 and 10, which are a list's: it trades one name and has no list around it. Its luck (1) and its t against
  holding the instrument (2) are judged against the tries on its instrument, and instead of 9 and 10:
  13. works on most of its market: of the other instruments of its run (the same strategy and timeframe on the rest of
      its list) at least half make money, three at least scored, as choosing on peers asks (`walkforward.MIN_PEERS`,
      `PEER_SHARE`): judged when read, from their results (`board.peers`), so a rule that pays on one chart only by
      luck shows it.
* A full re-run: from the page (Re-run in the header: every strategy on every list; with Single assets each strategy on
  each instrument alone too, then the ML task's picks; Everything also re-evaluates what the current code already
  produced) or `make rerun` (`SINGLE=1`, `EVERYTHING=1`, `RESUME=<run id>`). Some of it only, from a terminal: `make
  rerun SINGLE=1 EVERYTHING=1 LISTS="stockhunt_commodities"` (`scripts/rerun.py -u ... --names ... -t ...`, any of the
  three: `LISTS`, `NAMES`, `TFS`), for a market whose data changed: its lists as books, with SINGLE its instruments
  alone, then the same stages after the jobs as a full run (the picks read every result, the lists' figures every
  list's bars); the page names what such a run covers. A run fixes its jobs when it starts
  (run_job), shows its progress on the page and stops there at once: the evaluations running then start over when it
  resumes, everything finished stays saved, and resuming the same run does only what it had not done. Its log is
  logs/rerun_<id>.log. While it is on it keeps this Mac from sleeping (caffeinate; a closed lid still sleeps a laptop),
  and a restart of the server does not touch it. The jobs spread over the cores (`--workers`, `strategy_lab.batch`): the
  longest first, by the seconds each took when last saved, and the jobs on one list and timeframe in pieces on one
  process, which loads the list once for them; ml_direction fits its models in the main process meanwhile. After the
  jobs: the picks and the lists' figures (`python -m strategy_lab db lists`), both in the database; the server works
  the luck bars out again once the run has ended, the lists' and the single assets' (`python -m strategy_lab db
  judge`). A run with single assets removes the kept runs of a strategy on each instrument alone that its plan no
  longer makes, within the lists, strategies and timeframes it covers (`runs.unplanned_alone`: the rules on a list
  whose instruments its market's widest list holds today, a strategy no longer run alone), as an instrument that
  leaves a list goes with the list's next run; kept, they would read as stale for ever (1,000 of them after run 3,
  the ML task's lists on the instruments' ids of 2026-09-24/25). A run raises its processes' limit of open files to
  65,536 (`runs.OPEN_FILES`): exits walked on one-minute bars keep two memory-mapped files a name open, and a run
  started from the page is a child of the service, which launchd lets open 256 (20 list jobs of run 3 failed so). `scripts/week1.py` and `scripts/per_asset.py` run the list jobs or the single-asset jobs alone, as runs of
  their own.
* **STALE** on the board — the result was produced by code that has changed since (every result keeps a hash of the
  engine and strategy source it ran on). The page shows it faded, tagged "older code".
* The web UI (server/, ui/web): Research (a strategy on a list as one book, or on each asset alone) and a result's
  popup read the database through the server (`/api/research/...`, typed from ui/web/openapi.json: `make api` after the
  server's answers change); every result saved, luck bar worked out or evaluation code edited reaches an open page
  within a second (server-sent events). Portfolios is a mockup, a page with no data behind it
  (ui/mockup/portfolios.html), shown at /portfolios.

## Engine conventions

* A signal at a bar's close is filled at the next bar's open (`fill="next_open"`, how the firm's simulator trades);
  `fill="next_close"` is one bar more conservative.
* A position is the units its fill bought, held until its target moves, as an account holds it (the firm's simulator
  and products, and buy-and-hold here): a rule's positions each traded only when their own target moves (entry, exit,
  a resize of its share of the list), a panel strategy's book rebalanced whole when any weight moves (its monthly or
  weekly rebalance); between, a weight drifts with its price. Until 2026-09-27 every weight was traded back to its
  target on every bar, selling a trend's winners down each bar (SMA 20/200 on crypto Top-10 1d: 6.1% a year where
  held units make 10.2%, on stocks Top-10 1d 10.2% against 11.5%; IBS's short trades within 0.5% either way) and
  rebalancing every bar a book meant to be rebalanced monthly.
* No fill takes the positions held past the capital, as a cash account without margin buys. Held as units, the
  positions grow past their shares, and a trade entered at its share of the equity can need more money than is free
  (two seats of half: one bought and doubled holds two thirds, and half of the equity bought for the other would
  borrow a sixth): what a bar's fills add is then cut, pro rata, to the capital the positions it does not trade leave
  free, and a position bought short of its target keeps its units until its own target moves; a fill that reduces or
  closes a position always goes through. The drift itself is not cut: a short that goes against it grows past its
  share. The exposure a record holds (its figures' time in the market, and what the money test against buy & hold
  leaves in T-bills or borrows) is that of the positions held, not of the targets. Every fill taking its full share
  would have held up to 110-340% of the equity at the peaks of the trend rules' records (1d, crypto and stock Top-3 to
  Top-100; the capped records' CAGR -10 to +17% a year apart, drawdowns up to 18 points deeper), on money borrowed
  for nothing.
* A short is liquidated where its price reaches twice its average entry (`backtest.LIQUIDATION`; fills at the next
  open only, as the exits): closed at that price, inside a bar whose high reaches it or in the gap that passes it, so
  that it loses the money it was sold for and no more, as a position on its own margin at 1x does (the exchange's
  liquidation, its insurance fund taking what a gap carries past); it then stays out until its target leaves the short
  side (a book: until its next rebalance), and its trade is marked `liquidated`. Held as units without it, a short on
  a coin that multiplied lost more than the account (`big_move_follow` on crypto Top-100 1d, 2026-09-28: a drawdown of
  105%).
* Costs per side by asset class (`config.COSTS`): commission + half-spread; perp funding and short borrow are charged.
  A funding settlement is paid by the position held into its hour: Binance stamps 45% of them 1-10 ms after it. A spot
  quote of a commodity (gold, silver, platinum, palladium, WTI crude) pays half the spread its broker quoted at the
  fill instead of its class's half-spread (`engine.costs`), bar by bar: at a bar's open for a fill there, the median of
  its hours' spreads for a stop or a target filled inside it, at the close for a fill at the close; a bar with no
  quote of its own (the broker's daily pause, where the vendor prints one) the last spread quoted, and a bar before
  the first on record what the first year quoted at the same hour of the day, in dollars (the brokers set these
  spreads in dollars and seldom move them). Measured at the hours' close (`spread_refresh`), a month's median of the
  full spread: gold 3.9 bp in 2015-08 and 0.35 in 2025-04, platinum 54 bp in 2020-06 and 17 in 2026-08, palladium 203
  and 31, crude 14 and 1.7; a flat 2bp a side had been eleven times gold's half-spread of 2025-04, an eighth of
  palladium's of 2026-08 and a fiftieth of its 2020-06 one.
* Intrabar exits are checked at every one-minute bar inside a bar where the store has the instrument's minutes
  (`data.minutes`: crypto perps while a list holds them, the ML task's over their whole history, the FX majors since
  2003, stocks, ETFs and spot metals since 2020-03; CME futures have none), and against the bar's own high and low
  elsewhere: a stop hit inside a bar fills at its level, or at the open of the minute that gaps through it. Where a bar
  has minutes they alone decide: a stock's daily high or low beyond its hours' (a print of the consolidated tape the
  hourly bars do not carry: 2% of the Top-100's days by more than 10 bp) is not reached then. A trailing stop's level is
  re-set at each bar's close from the best high (low) of the bars since entry, as LeBeau's chandelier, or (`trail_every:
  minute`) at each minute's close, as a bot's stop following the price. A level holds still through a minute, so minute
  bars measure both exactly whatever the order of a minute's own high and low; a trail that follows every trade would
  need ticks (on crypto Top-10 1h the two ways of ordering a minute bracket it, and the stop re-set every minute did
  better than either). Which re-set earns more depends on the timeframe as the room does, so the trailing strategies'
  grids hold both for the walk-forward to choose. Out of sample on six lists (crypto Top-10 and Top-100, the stocks'
  Top-100, the ETFs, the FX majors, the spot metals) x three timeframes x the two strategies, against re-setting at each
  bar only, the choice gained where crypto trades intraday (1h and 4h: +0.75 pp a month at the median, 7 of 8 better,
  drawdowns 2.2 pp shallower, Sharpe +0.17), lost little on crypto 1d (-0.06 pp a month, Sharpe -0.02, where re-setting
  each minute alone lost 1.14 pp and 0.42) and left the other markets as they were (+0.00 pp a month and Sharpe at the
  median, within 0.2 pp either way). If the stop and the target are both touched in a minute (or a bar), the stop is
  assumed first; a gap through a level fills at the open, and no exit fills on a bar without a trade; after an exit the
  position stays flat until its target leaves that side (a rule's: until the rule turns, whatever its share of the
  capital does). A rule's exits are its trades' (`backtest.exited`), walked once from the trade's own entry: the
  positions its seats or slots hold close there, at those prices; walked again from each position's own fill, a name
  seated late was stopped out on its own (6.5-9.3% of `donchian_breakout`'s exits on 1d lists) while its seat stayed
  taken, empty, until the rule turned. In the check of a bar's delay the orders fill a bar later and a stop rests from
  that fill.
* A list that changes does not cut a rule's trade: a name that leaves keeps its seat until its rule ends the trade it
  holds (closes it, or turns to the other side, which is then not opened), and a name that joins waits for a free seat,
  the longest-waiting first and, of those that joined together, the more liquid first (as `slots` give theirs); there
  are as many seats as the list has names, so a seat's capital does not change (`strategy.seats`). An instrument that
  stopped trading is out of any list, a fixed one too: its seat and its share go to the others (a rule's position held
  after its last bar had kept its seat for ever: 24 dead seats on the stock Top-100, 1d). A panel strategy's exits are
  its own rebalances: it drops a name that leaves there (trade ledger: `left_list`). On the stock Top-100 with the
  buffer, 2% of a trend rule's trades run on after their name has left the list; without the buffer and seats a third of
  them were cut by the list. A rule that holds an exposure rather than trades (`@rule(exposure=True)`: always held, only
  resized; `vol_managed`) follows the list as a panel strategy does: kept to the end of its trade, a leaver would hold
  its seat for ever and no newcomer would be bought.
* An instrument cannot be traded on a bar it did not print, nor on one without an open, nor on one that shows no trade
  in a series that reports its volume (no volume, its open, high, low and close all at its last close: a thin ETF's
  quiet day, a halted stock's carried price, a coin's empty hour; `backtest._traded`): a fill due there waits for its
  next bar that trades and the position rides through the gap (a bar without an open, from the close before to its
  close). When an instrument is delisted its position is closed at the last close (trade
  ledger: `delisted`): a perp from the delisting date the exchange gives (`refresh binance-delistings`), any
  other series when its data stops more than a week before the rest of the panel; a series that is only a few days
  short with no delisting on record is a vendor lag and its position stays open to the end.
* The bar-by-bar parts are compiled with numba: the backtest's fills and valuation (walked instrument by instrument,
  as pandas keeps a frame's columns), the intrabar exit walk, the trade spans of the ledger, the rotations of random
  timing and the figures of each Monte Carlo resample (summed as numpy sums, so they come out as pandas gave them to
  the last bit); the first run in a fresh environment compiles them in a few seconds and caches the result next to
  the source.
* A rule's positions on an instrument are kept on disk (data/cache/positions/<code version>/, `strategy.fill_signals`)
  and read back by every later evaluation that gives the rule the same bars: the same rule on the same instrument is
  asked for on every list that holds it (Top-3 to Top-100, and the lists next to each for their checks) and on the
  instrument alone. The key is the code of the evaluation (every module and the rule's file, as a result's stamp, with
  the numpy, pandas and TA-Lib versions) and a hash of every field of every bar the rule is given: a change to either
  is a new entry, never a stale one. A version of the code that runs no more is never read again, and the whole
  directory can be deleted at any time: it costs only the rules' work again.
* Work a grid's signal configurations repeat on an instrument's bars is done once for them (`strategy_lab.shared`):
  the indicators and regimes a rule asks for on the bars or their columns (calm_trend's 108 configurations take one
  volatility regime, three confirmations of it and six moving averages). Each configuration gets its own copy, and an
  input is recognised by what it is (the bars, a column of them, a result handed out and not changed since), never
  by its values: anything else is computed as before, and every rule's positions come out the same bit for bit
  (tests/test_lookahead.py). A model grading a rule's trades keeps the features of their first bars by the
  instrument's bars and those bars (`trade_model`, the last 256 MB of them): another seed of the model and another
  list holding the name grade the same trades on the same bars.

## Data notes

* Every bar is stamped with the moment it closed (UTC). FX daily and all 4h bars are rebuilt from hourly bars (the
  vendor's FX dailies carry open == close on most 2022-2024 rows). Research timeframes: 1h, 4h and 1d.
* The FX majors' hourly history before the vendor's (2020-01) is Dukascopy's (`refresh dukascopy-extend`). Its
  AUD/USD candles of 2007-04-01..2008-09-20 and 2009-04-05..09-19 carry New York's wall clock written as UTC, four
  hours early in summer and five in winter, and are put on UTC (`refresh.DUKASCOPY_NEW_YORK_CLOCK`): as stamped, their
  Friday afternoons looked missing (97 daily bars), and the pair's moves lined up with EUR/USD's only four or five
  hours later (a return correlation of 0.35-0.73 shifted, 0.0-0.2 as stamped; Forexite's minutes match them to 2 bp
  shifted). The other series line up with one another as stamped, but all of them sat an hour late in their first
  months (checked 2026-09-28 against Forexite's minutes, whose clock the US payrolls confirm: their 12:30 UTC release
  is Forexite's widest minute, at 12:31, and fell in Dukascopy's hour closing at 14:00 in June and July 2003): the
  hours of EUR/USD, GBP/USD, USD/JPY and USD/CHF to 2003-07-25 and of gold to 2003-08-01 are put an hour earlier
  (`refresh.DUKASCOPY_HOUR_LATE`), and the pairs' week of 2003-07-28, late and on time in turn within a day, is not
  taken from Dukascopy (`refresh.DUKASCOPY_CLOCK_UNKNOWN`). The hours the pairs still lack (that week, the week's
  first hours of those AUD/USD weeks, days here and there) come from Forexite's one-minute bars where it has them
  (`refresh forexite-fill`: 1,230 hours, 1.4-2.3 bp from the stored closes at the median, and that week's 117-134 a
  pair, 1.1-1.6 bp).
* The ETFs' daily bars are decided day by day between Twelve Data and Sharadar (`refresh etf-daily`), neither right
  everywhere: Twelve Data's of 2008 are not the market's (DBC's and EEM's a single price a day with no volume all
  year, the sector ETFs' closes 50-140 bp off on 20-40 days while open, high and low agree to the cent) and it begins
  DBC and EEM in 2008 where Sharadar has them from their launch (2006-02, 2003-04); Sharadar's bond ETFs sit 5-40 bp
  under every venue's prices through most of July 2020 (SHY, IEF, LQD) and its FXI's highs under the venues' on days.
  A day they agree on to a price step is Sharadar's; a day they differ on is the one the venues' own daily bars bear
  out (Nasdaq, NYSE Arca, Cboe BZX, from 2018-05-01), before those Sharadar's, counted in the series' meta as
  unverified (Yahoo, the third source of the dividends, did not answer on 2026-09-27); a day only one has is that
  one's (SPY's of 1993-1997 Twelve Data's). A US listing's hourly session stays when its median close is within a
  price step of the day's range.
* History goes back to the vendor's first bar: daily stocks and ETFs from their listing or 1970 (AAPL 1980, SPY 1993),
  hourly stocks and ETFs from 2019-01-07, Twelve Data's first hour of US stocks (its ETFs' hours begin 2020-02-10: the
  year before is Databento's, below). The vendor's early daily
  history has holes of years in places (a few 1975 bars, then nothing until 1984): a series starts after its last
  hole longer than 10 days. A universe without an index membership therefore starts early and thin:
  `stockhunt_stocks` in 1970 with 19 names (100 from 1996), `etf_core` and `stockhunt_etfs` in 1993 with SPY alone
  until 1998. A list is scored only from the first day it holds half
  of its names (see What the numbers mean), and the week-1 run takes the desk's lists from the desk's own daily start
  (`strategy_lab.lists`: stocks 2003-01-02, coins 2020-08-01); elsewhere pass `start=` (`--start` on the command line).
* The vendor's hourly US-equity bars before 2020-06-29 are not where their labels say, and are re-stamped
  (`refresh.td_equity_hourly_starts`, checked against the vendor's daily and 15-minute bars): in 2019 they cover
  09:30-10:30 ... 15:30-16:00 as today; from 2020 to 2020-06-26 they cover clock hours (09:30-10:00, 10:00-11:00 ...
  15:00-16:00), so those sessions' 4h bars split at 13:00 instead of 13:30.
* The vendor's hourly series misses some corporate actions its daily series is adjusted for (APH's 2024 split, GE's
  and HON's spin-offs), and under a recycled ticker it serves another company (BNY before 2026-02, GEN before
  2021-09). A stock's hourly bars are therefore put on its daily bars' basis whenever they are written
  (`refresh.td_hourly_on_daily_basis`): a stretch at a steady price ratio is rescaled, another company's stretch is
  dropped with everything before it; the meta of the series records what was done.
* The sessions a stock's or ETF's hourly series lacks since 2019-01-07 are built from Databento's one-minute bars of
  its seven lit venues (Nasdaq, NYSE, NYSE Arca, Cboe BZX, Nasdaq BX and PSX, NYSE American; `refresh
  databento-hourly`): a member that left the market, the years before a ticker change (Twelve Data serves a renamed
  ticker only from its change: GAP from 2024-08-22, GPS before), another company's years under a ticker taken up
  again, a stretch its hourly bars do not match (HON before its 2026-06-29 split and spin-off), the ETFs' year before
  2020-02-10 and the days Twelve Data skips for nearly every stock (2019-12-31, 2020-01-02). Minutes are summed into
  the same 09:30-based hours, a minute's open and close from the venue that traded most in it; the session ends with
  its closing cross (the 16:00 print of the venue that traded most then, when more than any venue traded in the
  minute before), the other venues' 16:00 prints being trades after the close. Prices are put on the split-adjusted
  basis by Sharadar's split actions and each session's volume is scaled to the consolidated daily volume (the lit
  venues carry 50-67% of it). A session's last close is the daily close, the auction's price (the feeds miss it on
  some days, NYSE's in March-May 2020, and a minute's first print is not always the auction's), and its first open
  the daily open when that lies inside its first hour; a daily bar that only repeats the close before it (Sharadar's
  row for a day it lost the ticker: COHR 2022-09-08, VTRS 2020-11-17..19) leaves the session Databento's prices. The
  days around a ticker change are asked under both tickers (Sharadar's dates can be a week off the exchange's), and
  a day only the other ticker printed is kept when the daily bar's close lies in its range (after Twenty-First
  Century Fox became TFCF for its last days, FOX was the new Fox Corporation). Sessions with no trade on the lit
  venues stay without hourly bars: SMCI in 2019 (off Nasdaq, traded over the counter), a spin-off's when-issued
  weeks (CARR, OTIS, OGN, VNT), trading halts (SIVB, SBNY and FRC in 2023, BIIB during two FDA panels), and the
  zero-volume bar Sharadar adds on a delisting date. Hourly bars on days a company's daily series does not trade are
  another company's under the same ticker (Twelve Data's AGN, ARG and AET since 2020-2023) and are dropped.
* A US listing's hourly bars are held to its daily bars (`refresh.conform_to_daily`): within a regular session
  nothing trades above the day's high or below its low, the session opens at the day's open and closes at its close,
  and its volume is the day's. Twelve Data's hourly bars broke each of these: prints far outside the day (APA, HST and
  KDP on 2021-10-19 a third under the day's low; BALL's lows at 3.2 for 52 in 2023-03), a first bar holding the close
  before an earnings gap (FTNT 2025-08-07, SWKS 2025-02-06) or an IPO's offering price (CRWD, COIN), a first bar
  0.5% off the opening auction on 4.5% of the sessions, and 74-97% of the day's volume, a share that differs by name.
  So each bar is held within the day's range, the first opens at the day's open, the last closes at its close, and
  the session's volume is put on the day's; a session whose bars are not the day's prices at all (a split day left on
  the other basis: FAST on 2019-05-22 at twice the price, NBR's 1-for-50 day) or are off the session's grid is
  dropped, and Databento's minutes fill it (496 sessions of 53 names, 2026-09-26). A daily bar that only repeats the
  close before it holds no price, and its session stays as it is. A session whose bars show trading with no volume
  (Twelve Data put none at all in ~1% of the sessions of 2020-2022, and in 2019-2022 on some days the whole day's in
  one bar and none in the others, on 2022-05-27 for hundreds of members) takes the day's volume in the shape of the
  listing's other sessions, each hour its median share of the day.
* US stocks: every S&P 500 member since 1996 at the dates it was in the index, those that no longer trade included
  (Sharadar's bars as `sh:<ticker>`, from 1997-12-31: a stock list is scored from 1998); 9 members that left before
  1998 have no bars and each card names them. A Sharadar row that only carries the close before it forward on a day
  Twelve Data's bars show trading (Sharadar lost the ticker on a reorganisation or merger day) is Twelve Data's bar
  put on Sharadar's basis (`refresh.repair_sharadar_rows`), and a split row Sharadar left a day on the unadjusted basis
  is put on the adjusted one, reviewed case by case (BF.B 2018-02-28: `refresh.SHARADAR_SPLIT_DAY_LATE`; another split
  whose unadjusted close moves the next day is named in the log for the same review). Members whose data misses more than 2% of NYSE sessions while they
  were in the index are broken series and are excluded (listed in the card's notes).
* Liquidity ranks (`us_stocks_topN`, `crypto_topN`, `stockhunt_stocks`) use the median daily dollar volume over the
  last 60 (stocks) or 30 (crypto) calendar days, re-ranked at each month end; a day without trading counts as zero.
* Commodities are two markets, as on the firm's research desk. Their spot quotes (`stockhunt_commodities`: Twelve
  Data's `td:XAU/USD` ..., the prices the firm's simulator fills at) are thin for platinum, palladium and crude
  (palladium's daily returns correlate 0.96 with its future, platinum's 0.91, crude's 0.88, gold's and silver's 0.98;
  crude's spot rolls on its own dates) and have hourly bars from 2020 (gold's from 2003 and silver's from 2011 from
  Dukascopy: its silver quotes before 2011 are not the market's, a 2.3% spread and a mid 1.7% under the day's close in
  2005; crude's from 2019-02-18, Exness's): the list is scored on 1h and 4h from 2019-02-18 (gold, silver and crude;
  from 2020-01 until crude's hours were Exness's), on 1d from 1983-03 (gold, silver and crude). Crude's hours are not
  the vendor's but Exness's mid wherever it quotes, built from its ticks, and so are its minutes
  (`refresh.EXNESS_QUOTES`, `spread_refresh.exness_mid_minutes`): checked hour by hour against CME's
  crude future (2026-09-28), the vendor's crude mixes two prices a few percent apart within the hour from 2024-09 on
  and off, and in every month of 2025-05..10 and 2026-04..07 (its hourly returns correlate 0.41-0.80 with CME's there,
  Exness's 0.96-1.00, and swing back the next hour: an hourly model fading the swing made +451,665% out of sample,
  IBS +8.6% a month, and both lose on Exness's prices), while Exness's moves with CME's in every month it quotes and
  rolls with no jump. The vendor's hours stand only before Exness's first tick and after its last; the one hole in
  Exness's archive while CME traded, 2021-03-02 15:00..03-03 23:00 UTC, takes the vendor's returns on Exness's level
  (`refresh.EXNESS_HOLES`); a day of Exness's too short to be whole (a holiday's session) is made of its own hours,
  never the vendor's day, which quotes another contract month 1-5% apart for months of 2021-2022. A refresh of the
  vendor's crude (`refresh twelvedata`) fetches Exness's months not on disk and the one under way first, and their
  spreads, then adds the vendor's bars after the last stored one only: the stored ones are Exness's, which the
  vendor's do not match, and no restatement is looked for there. The metals' hours were screened hour by hour against
  CME's future of the same metal (2026-09-29, 2020-2026: the vendor's price over CME's more than 2% from that ratio's
  median over the day around it) and each one reviewed with Exness's mid beside it: 5 of silver's, 12 of platinum's
  and 26 of palladium's are not the market's where Exness's mid held to CME's (mostly a close the vendor left behind
  in a fast hour or before the daily pause and caught up the hour after: silver 2026-02-05 22:00 +1.60% into the hour
  against CME's -3.80% and Exness's -3.74%, then -5.08%; some a price no market made: palladium at 972-992 and
  894-897, flat, on 2025-06-27 and 06-30..07-01, and 1061 from 2025-09-01 19:00, a US holiday's close carried flat
  until the market opened), and take Exness's returns on the vendor's level, or have no bar where Exness did not quote
  them (`refresh.VENDOR_HOURS_WRONG`); the hours where Exness departed with the vendor (the spot's own moves against
  the futures, March-May 2020 above all) stay. Gold had none. The vendor's other hourly bars are first held to the
  vendor's daily bars (`refresh.conform_commodity_hourly`): nothing beyond the day's high or low, and a first open or
  last close beyond the day's range is the day's (silver's first hour of 2022-05-16 opened at 25.24 on a day that
  traded 20.84-21.72); an open or close inside the range stays as quoted (gold's and silver's hours before 2020 are
  Dukascopy's bid, 1-4 bp under the vendor's daily prices). Their days are then built from those hours wherever the
  hours make a day whole, as the FX majors' (`refresh._commodity_daily`): the vendor's own daily close is not the
  17:00 quote but a fixing's or a settlement's (platinum's, palladium's and crude's 25 bp and more from their last
  hour on a quarter to a third of the days since 2021; round figures such as platinum's 887.0 in 2019, often outside
  the day's quotes altogether), and its open often lies beyond its high or low (held within it). The vendor's days
  stand only before the hourly bars (gold's before 2003, silver's before 2011, platinum's and palladium's before
  2020, crude's before 2019-02, where the vendor's settlement closes about 0.8% under Exness's quote: its first day
  opens that far from the day before); Twelve Data now serves silver's back to 1982 (the store's began 2004-12, with
  2005 mostly missing), and a bar whose close lies beyond its own range is left out (platinum's 15 days of 2019,
  palladium's 4); crude's days follow NYMEX's holidays. Their CME futures
  (`cme_futures`: `cme:GC`
  gold, `SI` silver, `PL` platinum, `PA` palladium, `CL` WTI crude, `HG` copper, which has no spot quote at Twelve
  Data: its "HG1 Copper Spot" is a stock on Frankfurt's exchange; `refresh cme-hourly`) run from Databento's first day,
  2010-06-06: per product the most traded contract hour by hour, the older contracts put on the latest one's basis at
  each roll by the ratio of the two contracts' closes at the last hour both printed (a holder's returns, the roll's
  cost in them; no jump at a roll). They trade the FX week with a pause from 17:00 New York each day, and their day
  closes at 17:00 New York; the hour from 17:00 is left out (until 2015-09-17 the pause began at 17:15, and that
  quarter hour made a day of its own before a closed one). An hour without a trade has no bar (thin nights in
  platinum and palladium, holiday halts), so the 4h and daily bars are built from the hours that traded, each stamped
  at its end (`resample.futures_4h_from_1h`, `futures_1d_from_1h`). CME's trading days (`calendars.cme_days`) are its
  calendar's (Good Friday, Christmas and New Year closed, a short session on other US holidays) without six Monday
  holidays of 2012-2014 on which it did not open at all (Thanksgiving 2012 and 2013, Independence Day 2013, the
  Monday holidays of 2013 and early 2014); Databento has no trade in any contract from 2014-09-22 21:00 to 2014-09-26
  11:00 UTC (days it flags degraded), none in gold's two most traded contracts on 2012-09-12 and none of any COMEX
  contract (gold, silver, copper) on 2014-12-31, days it calls whole: holes no other source of futures fills. Cost:
  2bp a side (their own spreads are not on record: Databento's one-minute bid and ask of the six cost $44.18 from
  2010-06, beyond the credit left); the spot quotes pay their broker's spread (Engine conventions).
* Prices are split-adjusted, not dividend-adjusted: a US stock's or ETF's cash dividend (`refresh dividends`) is paid
  on its ex-date to the position held at the close before, strategies and buy-and-hold alike (a short pays it); one
  whose dividends were never fetched is paid none, with a warning.
