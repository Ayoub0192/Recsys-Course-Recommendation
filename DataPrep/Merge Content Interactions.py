import pandas as pd
import numpy as np
import os
import multiprocessing as mp

# =======================================
# CONFIG
# =======================================
INTERACTIONS_FILE = "merged_interactions.csv"
QUESTIONS_FILE = "questions.csv"
LECTURES_FILE = "lectures.csv"
OUTPUT_FILE = "c3rec_full_with_lectures.csv"
CHUNK_SIZE = 1_000_000  # 1M rows per chunk

print("Loading metadata (global init)...")

# =======================================
# LOAD GLOBAL METADATA ONCE (WINDOWS SAFE)
# =======================================
questions = pd.read_csv(QUESTIONS_FILE)
lectures = pd.read_csv(LECTURES_FILE)

questions["concept_id"] = (
    questions["tags"]
    .astype(str)
    .str.split(";")
    .str[0]
    .astype("int32")
)

lectures["concept_id"] = lectures["tags"].astype("int32")

# Globals shared with workers
question_meta = questions[["question_id", "concept_id", "correct_answer"]]
lecture_map = dict(zip(
    lectures["concept_id"].tolist(),
    lectures["lecture_id"].tolist()
))

print("Questions loaded:", len(questions))
print("Lectures loaded:", len(lectures))


# =======================================
# WORKER FUNCTION
# =======================================
def process_chunk(chunk):
    global question_meta, lecture_map

    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    chunk["elapsed_time"] = pd.to_numeric(chunk.get("elapsed_time"), errors="coerce")

    merged = chunk.merge(
        question_meta,
        on="question_id",
        how="left",
        copy=False
    )

    merged["correct"] = (merged["user_answer"] == merged["correct_answer"]).astype("Int64")

    merged["lecture_id_tmp"] = merged["concept_id"].map(lecture_map)

    valid_mask = merged["lecture_id_tmp"].notna()
    lec = merged[valid_mask].copy()

    lec["question_id"] = None
    lec["user_answer"] = None
    lec["correct"] = None
    lec["elapsed_time"] = None
    lec["lecture_id"] = lec["lecture_id_tmp"]
    lec["timestamp"] = lec["timestamp"] - 10_000

    lec = lec.drop(columns=["lecture_id_tmp"])
    merged = merged.drop(columns=["lecture_id_tmp"])

    merged["lecture_id"] = None

    final = pd.concat([lec, merged], ignore_index=True)

    final = final.sort_values(["user_id", "timestamp"], kind="mergesort")

    return final[
        ["user_id", "question_id", "timestamp",
         "user_answer", "correct", "elapsed_time",
         "concept_id", "lecture_id"]
    ]


# =======================================
# PARALLEL GENERATION
# =======================================
def parallel_generate():

    print("\nStarting parallel C3REC dataset generation...\n")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    header_written = False

    pool = mp.Pool(mp.cpu_count())
    print(f"Using {mp.cpu_count()} CPU cores.")

    chunk_reader = pd.read_csv(
        INTERACTIONS_FILE,
        chunksize=CHUNK_SIZE,
        engine="c"
    )

    for final_df in pool.imap(process_chunk, chunk_reader, chunksize=1):

        final_df.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="a",
            header=not header_written
        )
        header_written = True

        print("Chunk processed and written.")

    pool.close()
    pool.join()

    print("\n✔ DONE! Dataset saved to", OUTPUT_FILE)


# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    parallel_generate()
