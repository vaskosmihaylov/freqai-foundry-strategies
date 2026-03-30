#!/usr/bin/env python3
"""Summarize Freqtrade trade databases for local dry-run/live monitoring."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

TRADING_DAYS_PER_YEAR = 365.0
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TARGETS: dict[str, Path] = {
    "quickadapter": REPO_ROOT / "quickadapter/user_data/config.json",
    "reforcexy": REPO_ROOT / "ReforceXY/user_data/config.json",
}

DATE_COLUMNS = ("close_date", "close_date_utc", "open_date", "open_date_utc")
PROFIT_ABS_COLUMNS = ("close_profit_abs", "profit_abs", "realized_profit")
PROFIT_RATIO_COLUMNS = ("close_profit", "profit_ratio")
IS_OPEN_COLUMNS = ("is_open",)


@dataclass(frozen=True)
class Trade:
    """Closed trade data used for summary statistics."""

    opened_at: datetime
    closed_at: datetime
    profit_abs: float
    profit_ratio: float


@dataclass(frozen=True)
class Report:
    """Computed report values for a strategy database."""

    name: str
    database: Path
    total_trades: int
    wins: int
    losses: int
    draws: int
    avg_profit_pct: float
    total_profit_pct: float
    total_profit_abs: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy_pct: float
    max_drawdown_pct: float
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    cagr_pct: float | None
    trades_per_day: float | None
    avg_hold_hours: float


class TradeReportError(RuntimeError):
    """Raised when the trade database or config cannot be summarized."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Freqtrade trades from the local SQLite databases used by "
            "quickadapter and ReforceXY."
        )
    )
    parser.add_argument(
        "--strategy",
        choices=("all", "quickadapter", "reforcexy"),
        default="all",
        help="Strategy database to summarize. Defaults to both.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Explicit config path. When provided, this overrides --strategy and "
            "summarizes only the selected config."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Explicit SQLite database path. Only valid together with --config.",
    )
    return parser.parse_args()


def strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving string contents."""

    output: list[str] = []
    in_string = False
    escape = False
    in_block_comment = False
    in_line_comment = False

    i = 0
    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                output.append(char)
            i += 1
            continue

        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            i += 2
            continue
        if char == "/" and next_char == "/":
            in_line_comment = True
            i += 2
            continue
        if char == '"':
            in_string = True
        output.append(char)
        i += 1

    return "".join(output)


def normalize_jsonc_layout(text: str) -> str:
    """Repair common JSONC layout issues into valid JSON."""

    lines = [line.rstrip() for line in text.splitlines()]
    normalized_lines: list[str] = []
    previous: str | None = None

    def is_value_end(line: str) -> bool:
        stripped = line.rstrip()
        return stripped.endswith(("}", "]", '"')) or stripped.endswith(
            ("true", "false", "null")
        ) or stripped[-1:].isdigit()

    def starts_new_value(line: str) -> bool:
        stripped = line.lstrip()
        return (
            stripped.startswith('"')
            or stripped.startswith("{")
            or stripped.startswith("[")
            or stripped.startswith(("true", "false", "null"))
            or stripped[:1].isdigit()
            or stripped.startswith("-")
        )

    for current in lines:
        if not current.strip():
            continue
        if previous is None:
            previous = current
            continue

        if previous.rstrip().endswith(",") and current.lstrip().startswith(("}", "]")):
            previous = previous.rstrip().rstrip(",")

        if is_value_end(previous) and starts_new_value(current):
            stripped = current.lstrip()
            key_line = stripped.startswith('"') and ":" in stripped
            open_value_line = stripped.startswith(("{", "["))
            scalar_value_line = not key_line and (
                stripped.startswith('"')
                or stripped.startswith(("true", "false", "null"))
                or stripped[:1].isdigit()
                or stripped.startswith("-")
            )
            if key_line or open_value_line or scalar_value_line:
                previous = previous.rstrip()
                if not previous.endswith(","):
                    previous = previous + ","

        normalized_lines.append(previous)
        previous = current

    if previous is not None:
        if previous.rstrip().endswith(","):
            previous = previous.rstrip().rstrip(",")
        normalized_lines.append(previous)

    return "\n".join(normalized_lines)


def load_jsonc(path: Path) -> dict[str, Any]:
    stripped = strip_jsonc_comments(path.read_text(encoding="utf-8"))
    normalized = normalize_jsonc_layout(stripped)
    return json.loads(normalized)


def resolve_targets(args: argparse.Namespace) -> list[tuple[str, Path, Path | None]]:
    if args.config and not args.config.exists():
        raise TradeReportError(f"Config file not found: {args.config}")
    if args.db and not args.config:
        raise TradeReportError("--db can only be used together with --config")

    if args.config:
        config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
        database_path = None
        if args.db:
            database_path = args.db if args.db.is_absolute() else (Path.cwd() / args.db)
        return [(config_path.stem, config_path.resolve(), database_path.resolve() if database_path else None)]

    if args.strategy == "all":
        return [(name, path.resolve(), None) for name, path in DEFAULT_TARGETS.items()]

    return [(args.strategy, DEFAULT_TARGETS[args.strategy].resolve(), None)]


def parse_sqlite_path(db_url: str, config_path: Path) -> Path:
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        raise TradeReportError(f"Unsupported db_url in {config_path}: {db_url}")

    raw_path = Path(db_url.removeprefix(prefix))
    if raw_path.exists():
        return raw_path

    # Container configs use /freqtrade/user_data/<db>.sqlite. Resolve to the local user_data dir.
    if raw_path.name:
        candidate = config_path.parent / raw_path.name
        if candidate.exists() or candidate.parent.exists():
            return candidate

    raise TradeReportError(f"Could not resolve SQLite database from db_url: {db_url}")


def find_column(available: set[str], candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in available:
            return column
    return None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_closed_trades(database: Path) -> list[Trade]:
    if not database.exists():
        raise TradeReportError(f"Database file not found: {database}")

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(trades)").fetchall()
        }
        if not columns:
            raise TradeReportError(f"'trades' table not found in database: {database}")

        open_date_column = find_column(columns, ("open_date", "open_date_utc"))
        close_date_column = find_column(columns, ("close_date", "close_date_utc"))
        profit_abs_column = find_column(columns, PROFIT_ABS_COLUMNS)
        profit_ratio_column = find_column(columns, PROFIT_RATIO_COLUMNS)
        is_open_column = find_column(columns, IS_OPEN_COLUMNS)

        missing = [
            name
            for name, value in (
                ("open date", open_date_column),
                ("close date", close_date_column),
                ("profit_abs", profit_abs_column),
                ("profit_ratio", profit_ratio_column),
            )
            if value is None
        ]
        if missing:
            raise TradeReportError(
                f"Unsupported trades schema in {database}: missing {', '.join(missing)} column(s)"
            )

        query = (
            "SELECT "
            f"{open_date_column} AS open_date, "
            f"{close_date_column} AS close_date, "
            f"{profit_abs_column} AS profit_abs, "
            f"{profit_ratio_column} AS profit_ratio "
            "FROM trades "
            f"WHERE {close_date_column} IS NOT NULL"
        )
        if is_open_column:
            query += f" AND COALESCE({is_open_column}, 0) = 0"
        query += f" ORDER BY {close_date_column} ASC"

        trades: list[Trade] = []
        for row in connection.execute(query):
            opened_at = parse_timestamp(row["open_date"])
            closed_at = parse_timestamp(row["close_date"])
            if opened_at is None or closed_at is None:
                continue

            trades.append(
                Trade(
                    opened_at=opened_at,
                    closed_at=closed_at,
                    profit_abs=float(row["profit_abs"] or 0.0),
                    profit_ratio=float(row["profit_ratio"] or 0.0),
                )
            )

    if not trades:
        raise TradeReportError(f"No closed trades found in database: {database}")
    return trades


def load_starting_balance(config_path: Path, config: dict[str, Any]) -> float:
    dry_run_wallet = config.get("dry_run_wallet")
    if isinstance(dry_run_wallet, int | float):
        return float(dry_run_wallet)
    return 1000.0


def daterange(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def compute_max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown * 100.0


def compute_daily_returns(
    trades: list[Trade],
    starting_balance: float,
) -> tuple[list[float], list[float], float, float | None, float]:
    profits_by_day: dict[date, float] = defaultdict(float)
    for trade in trades:
        profits_by_day[trade.closed_at.date()] += trade.profit_abs

    dates = daterange(min(profits_by_day), max(profits_by_day))
    equity_curve = [starting_balance]
    daily_returns: list[float] = []
    equity = starting_balance
    for current_day in dates:
        previous_equity = equity
        equity += profits_by_day[current_day]
        equity_curve.append(equity)
        if previous_equity > 0:
            daily_returns.append((equity - previous_equity) / previous_equity)

    max_drawdown_pct = compute_max_drawdown_pct(equity_curve)

    elapsed_days = max((dates[-1] - dates[0]).days, 1)
    ending_balance = equity_curve[-1]
    cagr_pct: float | None = None
    if starting_balance > 0 and ending_balance > 0:
        cagr_pct = (
            ((ending_balance / starting_balance) ** (TRADING_DAYS_PER_YEAR / elapsed_days)) - 1.0
        ) * 100.0

    return daily_returns, equity_curve, max_drawdown_pct, cagr_pct, ending_balance


def compute_sharpe(daily_returns: list[float]) -> float | None:
    if len(daily_returns) < 2:
        return None
    volatility = pstdev(daily_returns)
    if volatility <= 0:
        return None
    return mean(daily_returns) / volatility * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_sortino(daily_returns: list[float]) -> float | None:
    downside_returns = [value for value in daily_returns if value < 0]
    if not downside_returns:
        return None
    downside_deviation = math.sqrt(mean([value * value for value in downside_returns]))
    if downside_deviation <= 0:
        return None
    return mean(daily_returns) / downside_deviation * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_report(name: str, config_path: Path, database_override: Path | None) -> Report:
    config = load_jsonc(config_path)
    starting_balance = load_starting_balance(config_path, config)
    database = database_override or parse_sqlite_path(str(config.get("db_url", "")), config_path)
    trades = load_closed_trades(database)

    total_trades = len(trades)
    wins = sum(1 for trade in trades if trade.profit_abs > 0)
    losses = sum(1 for trade in trades if trade.profit_abs < 0)
    draws = total_trades - wins - losses
    total_profit_abs = sum(trade.profit_abs for trade in trades)
    avg_profit_pct = mean(trade.profit_ratio for trade in trades) * 100.0
    total_profit_pct = total_profit_abs / starting_balance * 100.0 if starting_balance else 0.0
    win_rate_pct = wins / total_trades * 100.0 if total_trades else 0.0
    expectancy_pct = avg_profit_pct

    winning_profit = sum(trade.profit_abs for trade in trades if trade.profit_abs > 0)
    losing_profit = abs(sum(trade.profit_abs for trade in trades if trade.profit_abs < 0))
    profit_factor: float | None
    if losing_profit > 0:
        profit_factor = winning_profit / losing_profit
    elif winning_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = None

    daily_returns, _equity_curve, max_drawdown_pct, cagr_pct, _ending_balance = (
        compute_daily_returns(trades, starting_balance)
    )
    sharpe = compute_sharpe(daily_returns)
    sortino = compute_sortino(daily_returns)

    calmar: float | None = None
    if cagr_pct is not None and max_drawdown_pct > 0:
        calmar = cagr_pct / max_drawdown_pct

    first_open = min(trade.opened_at for trade in trades)
    last_close = max(trade.closed_at for trade in trades)
    elapsed_days = max((last_close - first_open).total_seconds() / 86400.0, 1e-9)
    trades_per_day = total_trades / elapsed_days if elapsed_days > 0 else None
    avg_hold_hours = mean(
        (trade.closed_at - trade.opened_at).total_seconds() / 3600.0 for trade in trades
    )

    return Report(
        name=name,
        database=database,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        draws=draws,
        avg_profit_pct=avg_profit_pct,
        total_profit_pct=total_profit_pct,
        total_profit_abs=total_profit_abs,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        expectancy_pct=expectancy_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        cagr_pct=cagr_pct,
        trades_per_day=trades_per_day,
        avg_hold_hours=avg_hold_hours,
    )


def format_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)


def render_table(reports: list[Report]) -> str:
    if not reports:
        return "No successful reports."

    headers = [
        "Strategy",
        "Trades",
        "Wins",
        "Losses",
        "Draws",
        "Win Rate %",
        "Avg Profit %",
        "ROI %",
        "PnL Abs",
        "Profit Factor",
        "Expectancy %",
        "Max DD %",
        "Sharpe",
        "Sortino",
        "Calmar",
        "CAGR %",
        "Trades/Day",
        "Avg Hold h",
    ]
    rows = [
        [
            report.name,
            str(report.total_trades),
            str(report.wins),
            str(report.losses),
            str(report.draws),
            format_number(report.win_rate_pct),
            format_number(report.avg_profit_pct),
            format_number(report.total_profit_pct),
            format_number(report.total_profit_abs),
            format_number(report.profit_factor),
            format_number(report.expectancy_pct),
            format_number(report.max_drawdown_pct),
            format_number(report.sharpe),
            format_number(report.sortino),
            format_number(report.calmar),
            format_number(report.cagr_pct),
            format_number(report.trades_per_day),
            format_number(report.avg_hold_hours),
        ]
        for report in reports
    ]
    widths = [
        max(len(str(cell)) for cell in [header, *[row[index] for row in rows]])
        for index, header in enumerate(headers)
    ]

    def render_row(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    output = [render_row(headers), separator]
    output.extend(render_row(row) for row in rows)
    return "\n".join(output)


def main() -> int:
    args = parse_args()
    reports: list[Report] = []
    errors: list[tuple[str, str]] = []
    for name, config_path, database_override in resolve_targets(args):
        try:
            reports.append(compute_report(name, config_path, database_override))
        except TradeReportError as error:
            errors.append((name, str(error)))

    if reports:
        print(render_table(reports))
        print()
        for report in reports:
            print(f"{report.name}: {report.database}")

    if errors:
        if reports:
            print()
        print("Errors:")
        for name, message in errors:
            print(f"- {name}: {message}")

    return 0 if reports else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TradeReportError as error:
        raise SystemExit(f"error: {error}") from error
