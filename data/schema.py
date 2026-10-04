# أضف إلى SCHEMA_SQL
EXTRA_SQL = """
CREATE TABLE IF NOT EXISTS risk_events (
    id BIGINT, ts TIMESTAMP DEFAULT current_timestamp,
    kind VARCHAR, severity VARCHAR, detail JSON
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGINT, ts TIMESTAMP DEFAULT current_timestamp,
    actor VARCHAR, action VARCHAR, payload JSON, result VARCHAR
);

CREATE TABLE IF NOT EXISTS proposals (
    id BIGINT, created_at TIMESTAMP DEFAULT current_timestamp,
    kind VARCHAR, title VARCHAR, body VARCHAR,
    status VARCHAR, score DOUBLE, metrics JSON, meta JSON
);
"""
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    epic        VARCHAR NOT NULL,
    resolution  VARCHAR NOT NULL,
    ts          TIMESTAMP NOT NULL,
    open        DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume      BIGINT,
    PRIMARY KEY (epic, resolution, ts)
);

CREATE TABLE IF NOT EXISTS signals (
    id BIGINT, epic VARCHAR, resolution VARCHAR, ts TIMESTAMP,
    strategy VARCHAR, side VARCHAR, strength DOUBLE, meta JSON
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id VARCHAR PRIMARY KEY, epic VARCHAR, strategy VARCHAR, side VARCHAR,
    entry_ts TIMESTAMP, entry_px DOUBLE, exit_ts TIMESTAMP, exit_px DOUBLE,
    size DOUBLE, pnl DOUBLE, pnl_net DOUBLE, costs DOUBLE, meta JSON
);

CREATE TABLE IF NOT EXISTS strategies (
    name VARCHAR PRIMARY KEY, version VARCHAR, spec JSON,
    created_at TIMESTAMP DEFAULT current_timestamp,
    status VARCHAR
);

CREATE TABLE IF NOT EXISTS proposals (
    id BIGINT, created_at TIMESTAMP DEFAULT current_timestamp,
    kind VARCHAR, title VARCHAR, body VARCHAR,
    status VARCHAR, score DOUBLE, meta JSON
);

CREATE TABLE IF NOT EXISTS research_notes (
    id BIGINT, ts TIMESTAMP DEFAULT current_timestamp,
    topic VARCHAR, body VARCHAR, tags VARCHAR
);
"""
