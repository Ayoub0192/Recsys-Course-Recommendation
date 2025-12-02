import polars as pl
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_FILE = "c3rec_full_with_lectures.csv"
OUTPUT_FILE = "c3rec_full_fixed.parquet"
CHUNK_SIZE = 2_000_000  # rows per batch

# stateful memory for user carry-over timestamps
last_ts = {}


def fix_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Monotonic timestamp fixing:
    - Vectorized cum_max per user inside the chunk (Rust)
    - Then carry-over per user across chunks (small Python dict)
    """
    global last_ts

    # Within-chunk monotonic timestamps per user (vectorized)
    fixed = df.with_columns([
        pl.col("timestamp")
        .cast(pl.Int64)
        .cum_max()          # <-- valid in your Polars
        .over("user_id")
        .alias("ts_local")
    ])

    # Cross-chunk carry-over correction
    ts = fixed["ts_local"].to_list()
    uid = fixed["user_id"].to_list()

    for i in range(len(ts)):
        prev = last_ts.get(uid[i], -1)
        if ts[i] < prev:
            ts[i] = prev + 1
        last_ts[uid[i]] = ts[i]

    return fixed.with_columns(pl.Series("timestamp", ts)).drop("ts_local")


# ----------------------------------------------------------
# Count rows for progress bar
# ----------------------------------------------------------
print("Counting rows...")
total_rows = sum(1 for _ in open(INPUT_FILE)) - 1
num_chunks = (total_rows // CHUNK_SIZE) + 1
print(f"Total rows: {total_rows:,}")
print(f"Expected chunks (approx): {num_chunks}")

# ----------------------------------------------------------
# Process CSV in batches and write with PyArrow
# ----------------------------------------------------------
print("\nStarting MAX-SPEED pipeline...")

reader = pl.read_csv_batched(INPUT_FILE, batch_size=CHUNK_SIZE)

writer = None
processed_batches = 0

with tqdm(total=num_chunks, desc="Fixing timestamps") as pbar:
    while True:
        batches = reader.next_batches(1)  # get 1 batch at a time

        if not batches:  # no more data
            break

        df = batches[0]  # this is already a Polars DataFrame
        fixed = fix_chunk(df)

        # Convert to Arrow table
        table = fixed.to_arrow()

        # Initialize writer on first batch
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)

        writer.write_table(table)

        processed_batches += 1
        pbar.update(1)

# Close writer
if writer is not None:
    writer.close()

print(f"\n✔ DONE — MAXIMUM SPEED MODE COMPLETE: {OUTPUT_FILE}")
print(f"✔ Batches processed: {processed_batches}")
