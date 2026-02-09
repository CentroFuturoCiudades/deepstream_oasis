# Writer service: consumes events from Azure Event Hub and writes to PostgreSQL
import os
import json
import time
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
import psycopg2
import psycopg2.pool
from azure.eventhub import EventHubConsumerClient

# -------------------------------------------------
# Load environment
# -------------------------------------------------
load_dotenv("writer_video.env")

EH_CONN = os.environ["EH_LISTEN_CONN_STR"]
CONSUMER_GROUP = os.getenv("EH_CONSUMER_GROUP", "writer")

PG_DSN = os.environ["PG_DSN"]

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
FLUSH_SECONDS = float(os.getenv("FLUSH_SECONDS", "1"))
STARTING_POSITION = os.getenv("STARTING_POSITION", "@latest")

# -------------------------------------------------
# PostgreSQL pool
# -------------------------------------------------
pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, dsn=PG_DSN)

# -------------------------------------------------
# SQL Insert Queries for each table
# -------------------------------------------------
INSERT_QUERIES = {
    "person_observed": None,
    "video_event": """
        INSERT INTO video_event (id, camera_id, start_ts, video_path, width, height)
        VALUES (%(id)s, %(camera_id)s, %(start_ts)s, %(video_path)s, %(width)s, %(height)s)
        ON CONFLICT (id) DO NOTHING
    """,
    "detection": """
        INSERT INTO detection (
            id, confidence, person_id, video_event_id, timestamp, 
            bbox, skeleton, px_geometry, real_geometry
        )
        VALUES (
            %(id)s, %(confidence)s, %(person_id)s, %(video_event_id)s, %(timestamp)s,
            %(bbox)s::jsonb, %(skeleton)s::jsonb, 
            ST_GeomFromText(%(px_geometry)s), ST_GeomFromText(%(real_geometry)s)
        )
        ON CONFLICT (id) DO NOTHING
    """,
}

PERSON_INSERT_SQL = """
    INSERT INTO person_observed (id)
    VALUES (%(id)s)
    ON CONFLICT (id) DO NOTHING
"""

PERSON_UPSERT_SQL = """
    INSERT INTO person_observed (id, age_group, confidence, model_version)
    VALUES (%(id)s, %(age_group)s, %(confidence)s, %(model_version)s)
    ON CONFLICT (id) DO UPDATE SET
        age_group = EXCLUDED.age_group,
        confidence = EXCLUDED.confidence,
        model_version = EXCLUDED.model_version
"""


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def prepare_row(data: dict) -> dict:
    """
    Prepare the row data for insertion.
    Converts nested dicts (bbox, skeleton) to JSON strings.
    """
    table = data.get("table")
    row = {k: v for k, v in data.items() if k != "table"}

    if table == "person_observed":
        row.pop("event", None)
        row.setdefault("age_group", None)
        row.setdefault("confidence", None)
        row.setdefault("model_version", None)
        return row

    elif table == "video_event":
        if "start_ts" in row and row["start_ts"]:
            ts = row["start_ts"]
            if not isinstance(ts, str):
                row["start_ts"] = datetime.utcnow().isoformat() + "Z"
        return row

    elif table == "detection":
        # Convert bbox and skeleton to JSON strings if they're dicts
        if "bbox" in row and row["bbox"] is not None:
            if isinstance(row["bbox"], dict):
                row["bbox"] = json.dumps(row["bbox"])
        else:
            row["bbox"] = None

        if "skeleton" in row and row["skeleton"] is not None:
            if isinstance(row["skeleton"], dict) or isinstance(row["skeleton"], list):
                row["skeleton"] = json.dumps(row["skeleton"])
        else:
            row["skeleton"] = None

        # Handle geometry fields (expecting WKT format or None)
        if "px_geometry" not in row or row["px_geometry"] is None:
            row["px_geometry"] = None
        if "real_geometry" not in row or row["real_geometry"] is None:
            row["real_geometry"] = None

        return row

    return row


# -------------------------------------------------
# Insert batch by table
# -------------------------------------------------
def insert_batch(table_batches: dict):
    """
    Insert batches of rows organized by table name.
    table_batches: { "video_event": [rows...], "detection": [rows...], ... }
    """
    if not table_batches:
        return 0

    total_inserted = 0
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                # Insert in order: video_event first, then person_observed, then detection
                # This respects foreign key constraints
                for table in ["video_event", "person_observed", "detection"]:
                    rows = table_batches.get(table, [])
                    if rows:
                        if table == "person_observed":
                            total_inserted += insert_person_rows(cur, rows)
                            continue

                        query = INSERT_QUERIES.get(table)
                        if query:
                            for row in rows:
                                try:
                                    cur.execute(query, row)
                                    total_inserted += 1
                                except Exception as e:
                                    print(f"❌ Insert error in '{table}': {e}")
                                    print(f"   Row data: {row}")
    finally:
        pool.putconn(conn)

    return total_inserted


def insert_person_rows(cur, rows) -> int:
    count = 0
    for row in rows:
        payload = {
            "id": row.get("id"),
            "age_group": row.get("age_group"),
            "confidence": row.get("confidence"),
            "model_version": row.get("model_version"),
        }
        is_classification = any(
            payload.get(field) is not None
            for field in ("age_group", "confidence", "model_version")
        )

        try:
            if is_classification:
                print("is_classification")
                cur.execute(PERSON_UPSERT_SQL, payload)
            else:
                cur.execute(PERSON_INSERT_SQL, {"id": payload["id"]})
            count += 1
        except Exception as e:
            print(f"❌ Insert error in 'person_observed': {e}")
            print(f"   Row data: {row}")
    return count


# -------------------------------------------------
# Event Hub consumer logic
# -------------------------------------------------
# Buffers organized by partition, then by table
buffers = defaultdict(lambda: defaultdict(list))
last_flush = defaultdict(lambda: time.time())


def on_event(partition_context, event):
    print("Received event")
    pid = partition_context.partition_id

    try:
        body = event.body_as_str(encoding="utf-8")
        data = json.loads(body)
    except Exception as e:
        print(f"[p{pid}] ❌ Invalid JSON: {e}")
        return

    table = data.get("table")
    if table not in INSERT_QUERIES:
        print(f"[p{pid}] ❌ Unknown table: {table}")
        return

    row = prepare_row(data)
    buffers[pid][table].append(row)

    # Count total items in buffer for this partition
    total_buffered = sum(len(rows) for rows in buffers[pid].values())

    now = time.time()
    if total_buffered >= BATCH_SIZE or (now - last_flush[pid]) >= FLUSH_SECONDS:
        try:
            count = insert_batch(dict(buffers[pid]))

            # Clear all table buffers for this partition
            for table_buf in buffers[pid].values():
                table_buf.clear()
            last_flush[pid] = now

            partition_context.update_checkpoint(event)
            print(f"[p{pid}] ✅ inserted {count} record(s) + checkpoint")

        except Exception as e:
            print(f"[p{pid}] ❌ DB insert failed — no checkpoint")
            print(e)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    print("==============================================")
    print(" EventHub → PostgreSQL Inferences Writer")
    print(" Tables: video_event", "person_observed", "detection")
    print(" Consumer group :", CONSUMER_GROUP)
    print(" Starting pos   :", STARTING_POSITION)
    print("==============================================")

    client = EventHubConsumerClient.from_connection_string(
        conn_str=EH_CONN,
        consumer_group=CONSUMER_GROUP,
    )

    with client:
        client.receive(
            on_event=on_event,
            starting_position=STARTING_POSITION,
        )


if __name__ == "__main__":
    main()
