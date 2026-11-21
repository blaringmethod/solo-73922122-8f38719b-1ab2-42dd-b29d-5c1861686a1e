"""Round history: atomic with the view commit, receipt lookup, real counts.

Covers the operator story behind /admin/rounds:

* a published view and its round record take effect together — a crash
  between the two cannot erase the publish from the history, and a history
  write failure rolls the view change back;
* the receipt's ``round_id`` finds the exact record, and the record's
  fetch/reuse counts and start time match the receipt;
* when two instances race on one revision, only the round that actually
  committed is marked ``published``;
* databases written before round receipts existed keep their view and
  history (columns are added in place).
"""

from __future__ import annotations

import sqlite3
import threading
import time

from app.db import Database
from tests.conftest import Stack


def _sync(stack: Stack, index: int = 0) -> dict:
    response = stack.api(index).post("/admin/sync")
    assert response.status_code == 200, response.text
    return response.json()


def _rounds(stack: Stack, index: int = 0, **params) -> list[dict]:
    response = stack.api(index).get("/admin/rounds", params=params)
    assert response.status_code == 200, response.text
    return response.json()["rounds"]


def _transfers_view(stack: Stack, index: int = 0) -> dict:
    response = stack.api(index).get("/transfers", params={"address": "addr-alice"})
    assert response.status_code == 200, response.text
    return response.json()


def _cache_head_block(stack: Stack) -> None:
    """Simulate a crash artifact: the new head block fetched but unpublished."""
    head = stack.source_client().get("/head").json()
    block = stack.source_client().get(f"/blocks/{head['hash']}").json()
    from app.db import BlockRecord, TransferRecord

    db = stack.api_servers[0].server.config.app.state.db
    db.cache_block(
        BlockRecord(
            hash=block["hash"],
            height=block["height"],
            parent_hash=block["parent_hash"],
            transfers=tuple(
                TransferRecord(
                    txid=t["txid"],
                    from_address=t["from_address"],
                    to_address=t["to_address"],
                    amount=t["amount"],
                )
                for t in block["transfers"]
            ),
        ),
        head["revision"],
    )


# --------------------------------------------------------------------- receipt
def test_committed_round_is_findable_by_receipt_round_id(synced: Stack):
    stack = synced
    stack.source_client().post("/admin/advance", json={"count": 2})
    receipt = _sync(stack)
    assert receipt["outcome"] == "committed"
    assert receipt["published"] is True
    assert receipt["fetched_blocks"] == 2

    # The receipt's round_id finds exactly one record, and it agrees with
    # the receipt on counts, times, instance and observed revision.
    rows = _rounds(stack, round_id=receipt["round_id"])
    assert len(rows) == 1
    row = rows[0]
    assert row["round_id"] == receipt["round_id"]
    assert row["instance_id"] == receipt["instance_id"]
    assert row["outcome"] == "committed"
    assert row["published"] == 1
    assert row["fetched_blocks"] == receipt["fetched_blocks"]
    assert row["reused_blocks"] == receipt["reused_blocks"]
    assert row["changed_blocks"] == receipt["changed_blocks"]
    assert row["started_at"] == receipt["started_at"]
    assert row["finished_at"] >= row["started_at"]
    assert row["observed_revision"] == receipt["observed_head"]["revision"]
    assert row["observed_hash"] == receipt["observed_head"]["hash"]

    # "Which instance published the view?" — the revision filter answers it.
    by_revision = _rounds(
        stack, revision=receipt["observed_head"]["revision"], published="true"
    )
    assert [r["round_id"] for r in by_revision] == [receipt["round_id"]]
    assert by_revision[0]["instance_id"] == "instance-a"

    # An unknown receipt id finds nothing, never a wrong row.
    assert _rounds(stack, round_id="does-not-exist") == []


def test_stale_round_is_recorded_with_actual_times_and_counts(synced: Stack):
    stack = synced
    receipt = _sync(stack)  # same revision: stale before any fetch
    assert receipt["outcome"] == "stale"
    assert receipt["published"] is False

    rows = _rounds(stack, round_id=receipt["round_id"])
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "stale"
    assert row["published"] == 0
    assert row["fetched_blocks"] == receipt["fetched_blocks"] == 0
    assert row["reused_blocks"] == receipt["reused_blocks"] == 0
    assert row["started_at"] == receipt["started_at"]
    assert row["finished_at"] >= row["started_at"]


# ------------------------------------------------------------------- atomicity
def test_history_write_failure_rolls_back_the_view_change(synced: Stack):
    stack = synced
    before = _transfers_view(stack)
    stack.source_client().post("/admin/advance", json={"count": 2})

    # Break the history sink: any insert into ingest_rounds aborts.
    conn = sqlite3.connect(stack.db_path, timeout=10)
    conn.execute(
        "CREATE TRIGGER block_round_history BEFORE INSERT ON ingest_rounds "
        "BEGIN SELECT RAISE(ABORT, 'history sink broken'); END"
    )
    conn.commit()
    try:
        receipt = _sync(stack)
        assert receipt["outcome"] == "failed"
        assert "history sink broken" in receipt["error"]

        # The view commit was rolled back together with its history record:
        # the previously published view is served unchanged.
        after = _transfers_view(stack)
        assert after["chain_view"] == before["chain_view"]
        assert after["head_basis"] == before["head_basis"]
        status = stack.api().get("/chain/status").json()
        assert status["serving"] is True
        assert status["published_revision"] == before["head_basis"]["revision"]
        # And no half-written round row survived the rollback.
        assert _rounds(stack, round_id=receipt["round_id"]) == []
    finally:
        conn.execute("DROP TRIGGER block_round_history")
        conn.commit()
        conn.close()

    # With the sink repaired the very next round commits and is auditable.
    receipt = _sync(stack)
    assert receipt["outcome"] == "committed"
    rows = _rounds(stack, round_id=receipt["round_id"])
    assert len(rows) == 1 and rows[0]["published"] == 1
    assert _transfers_view(stack)["head_basis"]["height"] == 7


def test_failed_round_records_actual_counts_and_times(synced: Stack):
    stack = synced
    stack.source_client().post("/admin/advance", json={"count": 3})
    _cache_head_block(stack)  # head block already cached: reused, not fetched
    stack.source_client().post("/admin/faults", json={"block_failures": 1})
    try:
        receipt = _sync(stack)
    finally:
        stack.source_client().post("/admin/faults", json={})
    assert receipt["outcome"] == "failed"
    assert receipt["reused_blocks"] == 1  # the cached head block
    assert receipt["fetched_blocks"] == 0  # the first real fetch failed

    rows = _rounds(stack, round_id=receipt["round_id"])
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "failed"
    assert row["published"] == 0
    # Regression: a mid-pull failure used to log both counters as zero.
    assert row["reused_blocks"] == 1
    assert row["fetched_blocks"] == 0
    assert row["error"]
    assert row["started_at"] == receipt["started_at"]
    assert row["finished_at"] >= row["started_at"]
    assert row["observed_revision"] == receipt["observed_head"]["revision"]


def test_round_record_survives_restart(synced: Stack):
    stack = synced
    stack.source_client().post("/admin/advance", json={"count": 2})
    receipt = _sync(stack)
    assert receipt["outcome"] == "committed"

    stack.api_servers[0].stop()
    stack.api_servers.clear()
    stack.add_api("instance-restarted")

    rows = _rounds(stack, round_id=receipt["round_id"])
    assert len(rows) == 1
    assert rows[0]["published"] == 1
    assert rows[0]["instance_id"] == "instance-a"
    assert rows[0]["outcome"] == "committed"


# ------------------------------------------------------------------ two writers
def test_only_the_committing_round_is_marked_published(synced: Stack):
    """A slow pull by one instance must not stop the other from syncing, and
    when both process the same revision only the winner is recorded as the
    publisher."""
    stack = synced
    stack.add_api("instance-b")
    stack.source_client().post("/admin/advance", json={"count": 2})

    # Instance A's source client becomes slow on block pulls; instance B
    # keeps the fast one.  A's round is still pulling when B commits.
    app_a = stack.api_servers[0].server.config.app.state
    real_source = app_a.ingestor.source

    class SlowBlockSource:
        def get_head(self):
            return real_source.get_head()

        def get_block(self, block_hash):
            time.sleep(0.4)
            return real_source.get_block(block_hash)

    app_a.ingestor.source = SlowBlockSource()

    slow_receipt: dict = {}

    def slow_round():
        slow_receipt.update(_sync(stack, index=0))

    thread = threading.Thread(target=slow_round)
    thread.start()
    time.sleep(0.2)  # A is now stuck inside its first slow block pull

    fast_receipt = _sync(stack, index=1)
    assert thread.is_alive(), "instance B could not sync while A was pulling"
    thread.join(15)

    assert fast_receipt["outcome"] == "committed"
    assert fast_receipt["published"] is True
    assert slow_receipt["outcome"] == "stale"
    assert slow_receipt["published"] is False
    # Both rounds observed the same source revision.
    revision = fast_receipt["observed_head"]["revision"]
    assert slow_receipt["observed_head"]["revision"] == revision

    rows = _rounds(stack, revision=revision)
    assert len(rows) == 2
    published_rows = [r for r in rows if r["published"] == 1]
    assert len(published_rows) == 1, "both instances were recorded as publisher"
    assert published_rows[0]["instance_id"] == "instance-b"
    assert published_rows[0]["round_id"] == fast_receipt["round_id"]

    loser = next(r for r in rows if r["published"] == 0)
    assert loser["instance_id"] == "instance-a"
    assert loser["outcome"] == "stale"
    assert loser["round_id"] == slow_receipt["round_id"]
    # The loser's record matches its own receipt, whatever work it did.
    assert loser["fetched_blocks"] == slow_receipt["fetched_blocks"]
    assert loser["reused_blocks"] == slow_receipt["reused_blocks"]
    assert loser["started_at"] == slow_receipt["started_at"]

    # Both instances serve the view B committed.
    assert (
        _transfers_view(stack, index=0)["chain_view"]
        == _transfers_view(stack, index=1)["chain_view"]
    )


# -------------------------------------------------------------------- migration
OLD_SCHEMA = """
CREATE TABLE blocks (
    hash TEXT PRIMARY KEY,
    height INTEGER NOT NULL,
    parent_hash TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    source_revision INTEGER
);
CREATE TABLE block_transfers (
    txid TEXT PRIMARY KEY,
    block_hash TEXT NOT NULL REFERENCES blocks(hash),
    from_address TEXT NOT NULL,
    to_address TEXT NOT NULL,
    amount INTEGER NOT NULL
);
CREATE TABLE chain_blocks (
    height INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE REFERENCES blocks(hash),
    parent_hash TEXT NOT NULL
);
CREATE TABLE sync_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    published_revision INTEGER NOT NULL DEFAULT 0,
    head_height INTEGER NOT NULL DEFAULT -1,
    head_hash TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'empty',
    published_at TEXT,
    last_round_at TEXT,
    last_error TEXT,
    last_error_at TEXT,
    rounds INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO sync_state (id) VALUES (1);
CREATE TABLE ingest_lease (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    owner TEXT,
    expires_at REAL
);
INSERT OR IGNORE INTO ingest_lease (id, owner, expires_at) VALUES (1, NULL, 0);
CREATE TABLE ingest_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    observed_revision INTEGER,
    observed_height INTEGER,
    observed_hash TEXT,
    ancestor_height INTEGER,
    fetched_blocks INTEGER NOT NULL DEFAULT 0,
    reused_blocks INTEGER NOT NULL DEFAULT 0,
    changed_blocks INTEGER NOT NULL DEFAULT 0,
    removed_blocks INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL,
    error TEXT
);
"""

GENESIS = "aa" * 32
BLOCK1 = "bb" * 32
BLOCK2 = "cc" * 32


def _seed_old_database(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    for height, (block_hash, parent) in enumerate(
        [(GENESIS, "0" * 64), (BLOCK1, GENESIS), (BLOCK2, BLOCK1)]
    ):
        conn.execute(
            "INSERT INTO blocks (hash, height, parent_hash, first_seen_at, "
            "source_revision) VALUES (?, ?, ?, ?, ?)",
            (block_hash, height, parent, "2026-01-01T00:00:00Z", 7),
        )
        conn.execute(
            "INSERT INTO chain_blocks (height, hash, parent_hash) VALUES (?, ?, ?)",
            (height, block_hash, parent),
        )
    conn.execute(
        "UPDATE sync_state SET published_revision = 7, head_height = 2, "
        "head_hash = ?, status = 'ok', published_at = '2026-01-01T00:00:01Z', "
        "rounds = 3 WHERE id = 1",
        (BLOCK2,),
    )
    conn.execute(
        "INSERT INTO ingest_rounds "
        "(instance_id, started_at, finished_at, observed_revision, "
        "observed_height, observed_hash, ancestor_height, fetched_blocks, "
        "reused_blocks, changed_blocks, removed_blocks, outcome, error) "
        "VALUES ('old-instance', '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z', "
        "7, 2, ?, NULL, 3, 0, 3, 0, 'committed', NULL)",
        (BLOCK2,),
    )
    conn.commit()
    conn.close()


def test_old_database_is_migrated_preserving_view_and_history(tmp_path):
    path = tmp_path / "old.db"
    _seed_old_database(path)

    db = Database(str(path))
    db.initialize()  # must migrate in place, not recreate

    # The published view survived.
    state = db.get_state()
    assert state.published_revision == 7
    assert state.head_height == 2
    assert state.head_hash == BLOCK2
    snapshot = db.snapshot()
    assert [b["height"] for b in snapshot.blocks] == [0, 1, 2]
    assert snapshot.blocks[2]["hash"] == BLOCK2

    # The historical row survived; a pre-receipt committed round counts as
    # published, and it simply has no round_id.
    rows = db.recent_rounds()
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "old-instance"
    assert rows[0]["outcome"] == "committed"
    assert rows[0]["published"] == 1
    assert rows[0]["round_id"] is None
    assert rows[0]["fetched_blocks"] == 3

    # New rounds are recorded alongside the old ones, with receipt ids.
    db.log_round(
        round_id="new-round-1",
        instance_id="instance-a",
        started_at="2026-01-02T00:00:00Z",
        observed=None,
        ancestor_height=None,
        fetched_blocks=0,
        reused_blocks=0,
        changed_blocks=0,
        removed_blocks=0,
        outcome="failed",
        error="source unreachable",
    )
    rows = db.recent_rounds()
    assert len(rows) == 2
    assert {r["round_id"] for r in rows} == {"new-round-1", None}
    found = db.recent_rounds(round_id="new-round-1")
    assert len(found) == 1 and found[0]["outcome"] == "failed"
