from pathlib import Path
import sys
import time

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "final_dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "generation"

OUTPUT_FILE = OUTPUT_DIR / "generated_replies.xlsx"
METADATA_FILE = OUTPUT_DIR / "generation_metadata.txt"

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"


# Generation settings

SYSTEM_PROMPT = (
    "Write the next reply to the conversation. Generate a natural,\n"
    "contextually appropriate social-media reply, as an ordinary Reddit\n"
    "participant would write it.\n\n"
    "Match the tone, formality, conversational style, and level of detail of\n"
    "the preceding conversation. When the conversation is casual, informal,\n"
    "humorous, sarcastic, or brief, respond in a similarly natural style.\n"
    "When a more serious or substantive response is appropriate, respond\n"
    "accordingly.\n\n"
    "Do not sound like an assistant, customer-service agent, or formal\n"
    "language model.\n\n"
    'Output exactly one reply and nothing else. Do not continue the\n'
    'conversation with additional speaker turns. Do not write speaker labels\n'
    'such as "Speaker A:" or "Speaker B:". Do not provide an explanation,\n'
    "analysis, commentary, or alternatives. Do not mention these instructions,\n"
    "an AI system, or a language model."
)

TEMPERATURE = 0.7
TOP_P = 0.8
SEED = 2026
NUM_PREDICT = 128
THINK = False


EXPECTED_COLUMNS = [
    "Sr.No.",
    "Conversation ID",
    "Condition",
    "Context",
]

OUTPUT_COLUMNS = EXPECTED_COLUMNS + ["Generated Reply"]

def fail(message):
    # Raise a RuntimeError with the given message
    raise RuntimeError(message)


def normalize_for_validation(value):
    if pd.isna(value):
        return ""
    return str(value)


# Input checks
def validate_input_dataset(df):

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        fail(
            "Input Excel has unexpected columns.\n"
            f"Expected exactly: {EXPECTED_COLUMNS}\n"
            f"Found: {actual_columns}"
        )

    if len(df) != 100:
        fail(
            f"Expected exactly 100 input rows, found {len(df)}."
        )

    # 50 conversation IDs, each appearing twice
    conversation_ids = df["Conversation ID"].tolist()

    if len(set(conversation_ids)) != 50:
        fail(
            "Expected exactly 50 unique Conversation IDs, found "
            f"{len(set(conversation_ids))}."
        )

    counts = df["Conversation ID"].value_counts()

    bad_counts = counts[counts != 2]
    if not bad_counts.empty:
        fail(
            "Every Conversation ID must occur exactly twice. "
            f"Invalid IDs: {bad_counts.to_dict()}"
        )

    # 50 A rows, 50 B rows, nothing else
    conditions = df["Condition"].tolist()

    if conditions.count("A") != 50:
        fail(
            f"Expected exactly 50 Condition A rows, found "
            f"{conditions.count('A')}."
        )

    if conditions.count("B") != 50:
        fail(
            f"Expected exactly 50 Condition B rows, found "
            f"{conditions.count('B')}."
        )

    unexpected_conditions = sorted(
        set(conditions) - {"A", "B"}
    )

    if unexpected_conditions:
        fail(
            "Unexpected condition values found: "
            f"{unexpected_conditions}"
        )

    # one A row and one B row per conversation
    for conversation_id, group in df.groupby(
        "Conversation ID", sort=False
    ):
        group_conditions = list(group["Condition"])

        if group_conditions.count("A") != 1:
            fail(
                f"Conversation ID {conversation_id!r} does not have "
                "exactly one Condition A row."
            )

        if group_conditions.count("B") != 1:
            fail(
                f"Conversation ID {conversation_id!r} does not have "
                "exactly one Condition B row."
            )

    # context must be non-empty and use the right Speaker A/B format
    for row_number, row in df.iterrows():
        condition = normalize_for_validation(row["Condition"])
        context = normalize_for_validation(row["Context"])

        if not context:
            fail(
                f"Row {row_number + 2}: Context is empty."
            )

        if condition == "A":
            if not context.startswith("Speaker B: "):
                fail(
                    f"Row {row_number + 2}: Condition A context does not "
                    'start with "Speaker B: ".'
                )

        elif condition == "B":
            if not context.startswith("Speaker A: "):
                fail(
                    f"Row {row_number + 2}: Condition B context does not "
                    'start with "Speaker A: ".'
                )

            if "\nSpeaker B: " not in context:
                fail(
                    f"Row {row_number + 2}: Condition B context does not "
                    'contain "\\nSpeaker B: ".'
                )

    # the parent text has to be identical in A and B
    for conversation_id, group in df.groupby(
        "Conversation ID", sort=False
    ):
        row_a = group[group["Condition"] == "A"].iloc[0]
        row_b = group[group["Condition"] == "B"].iloc[0]

        context_a = normalize_for_validation(row_a["Context"])
        context_b = normalize_for_validation(row_b["Context"])

        parent_a = context_a[len("Speaker B: "):]

        speaker_b_marker = "\nSpeaker B: "

        if speaker_b_marker not in context_b:
            fail(
                f"Conversation ID {conversation_id!r}: Condition B is "
                "missing the Speaker B turn."
            )

        parent_b = context_b.split(
            speaker_b_marker, 1
        )[1]

        if parent_a != parent_b:
            fail(
                f"Conversation ID {conversation_id!r}: Parent text differs "
                "between Condition A and Condition B."
            )


    forbidden_target_columns = {
        "Human Target",
        "human_target",
        "Target",
        "target",
    }

    found_target_columns = (
        forbidden_target_columns.intersection(actual_columns)
    )

    if found_target_columns:
        fail(
            "Human target column detected in input workbook: "
            f"{sorted(found_target_columns)}"
        )

    print("Input validation = PASSED")

def verify_ollama_model():
    

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10,
        )
    except requests.RequestException as exc:
        fail(
            "Could not connect to Ollama at "
            f"{OLLAMA_ENDPOINT}\n"
            f"Error: {exc}"
        )

    if response.status_code != 200:
        fail(
            "Ollama responded with HTTP "
            f"{response.status_code} while checking installed models."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        fail(
            "Ollama returned invalid JSON while checking installed models: "
            f"{exc}"
        )

    installed_models = []

    for model_info in payload.get("models", []):
        name = model_info.get("name")
        if name:
            installed_models.append(name)

    if MODEL not in installed_models:
        fail(
            f"Required Ollama model is not available:\n{MODEL}\n\n"
            "Installed models:\n"
            + "\n".join(installed_models)
        )

    print(f"Ollama model check = PASSED ({MODEL})")


def generate_one(context):

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": context,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
        },
        "think": THINK,
    }

    response = requests.post(
        OLLAMA_ENDPOINT,
        json=payload,
        timeout=600,
    )

    if response.status_code != 200:
        fail(
            f"Ollama generation request failed with HTTP "
            f"{response.status_code}: {response.text}"
        )

    try:
        result = response.json()
    except ValueError as exc:
        fail(
            f"Ollama returned invalid JSON: {exc}"
        )

    if "response" not in result:
        fail(
            "Ollama response did not contain the expected 'response' field."
        )

    generated_reply = result["response"]

    if not isinstance(generated_reply, str):
        fail(
            "Ollama 'response' field was not a string."
        )

    if generated_reply == "":
        fail(
            "Ollama returned an empty response."
        )

    return generated_reply


# Metadata file
def write_metadata(
    successful_generations,
    failed_generations
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = f"""FINAL QWEN GENERATION

Model: {MODEL}
Ollama endpoint: {OLLAMA_ENDPOINT}

Frozen system prompt:
{SYSTEM_PROMPT}

Temperature = {TEMPERATURE}
Top-p = {TOP_P}
Seed = {SEED}
num_predict = {NUM_PREDICT}
Thinking = {str(THINK).lower()}
Stop sequences = none

Number of conversations = 50
Number of input rows = 100
Number of generations = 100
Condition A generations = 50
Condition B generations = 50

Input file: {INPUT_FILE}
Output file: {OUTPUT_FILE}

Successful generations: {successful_generations}
Failed generations: {failed_generations}
"""

    METADATA_FILE.write_text(
        metadata,
        encoding="utf-8",
    )

def main():

    print("=" * 70)
    print("FINAL QWEN GENERATION")
    print("=" * 70)

    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL}")

    if not INPUT_FILE.exists():
        fail(
            f"Frozen input Excel file does not exist:\n{INPUT_FILE}"
        )

    if not INPUT_FILE.is_file():
        fail(
            f"Input path is not a file:\n{INPUT_FILE}"
        )

    try:
        df = pd.read_excel(
            INPUT_FILE,
            engine="openpyxl",
        )
    except Exception as exc:
        fail(
            f"Could not read frozen input Excel file:\n"
            f"{INPUT_FILE}\n"
            f"Error: {exc}"
        )

    # run all input checks before calling the model
    validate_input_dataset(df)

    # and check that Ollama is up
    verify_ollama_model()

    # generation is the only folder this script writes to
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # one reply per row; stop at the first failure
    generated_replies = []

    successful_generations = 0
    failed_generations = 0

    start_total = time.perf_counter()

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        sr_no = row["Sr.No."]
        conversation_id = row["Conversation ID"]
        condition = row["Condition"]
        context = row["Context"]

        print(
            f"[{position}/100] "
            f"Sr.No.={sr_no} | "
            f"Conversation ID={conversation_id} | "
            f"Condition={condition}"
        )

        generation_start = time.perf_counter()

        try:
            generated_reply = generate_one(context)
        except Exception as exc:
            failed_generations += 1

            write_metadata(
                successful_generations=successful_generations,
                failed_generations=failed_generations,
            )

            print()
            print("=" * 70)
            print("Generation status = FAILED")
            print("=" * 70)
            print(f"Failed row: {position}")
            print(f"Sr.No.: {sr_no}")
            print(f"Conversation ID: {conversation_id}")
            print(f"Condition: {condition}")
            print(f"Reason: {exc}")
            print()
            print(
                "Experiment stopped. No final generated_replies.xlsx "
                "was written."
            )

            return 1

        generation_time = time.perf_counter() - generation_start

        # keep the reply exactly as the model returned it
        generated_replies.append(generated_reply)

        successful_generations += 1

        print(
            f"  Success "
            f"({generation_time:.3f} sec)"
        )

    total_generation_time = (
        time.perf_counter() - start_total
    )

    # sanity checks before saving
    if len(generated_replies) != 100:
        fail(
            "Internal generation count error: expected 100 responses, "
            f"collected {len(generated_replies)}."
        )

    if successful_generations != 100:
        fail(
            "Generation did not complete successfully: "
            f"{successful_generations}/100."
        )

    output_df = df.copy()
    output_df["Generated Reply"] = generated_replies

    if list(output_df.columns) != OUTPUT_COLUMNS:
        fail(
            "Output schema is incorrect:\n"
            f"Expected: {OUTPUT_COLUMNS}\n"
            f"Found: {list(output_df.columns)}"
        )

    if len(output_df) != 100:
        fail(
            "Output row count is incorrect."
        )

    output_df.to_excel(
        OUTPUT_FILE,
        index=False,
        engine="openpyxl",
    )

    try:
        persisted_df = pd.read_excel(
            OUTPUT_FILE,
            engine="openpyxl",
        )
    except Exception as exc:
        fail(
            f"Final output workbook could not be reopened:\n{exc}"
        )

    if list(persisted_df.columns) != OUTPUT_COLUMNS:
        fail(
            "Persisted output workbook has an unexpected schema."
        )

    if len(persisted_df) != 100:
        fail(
            "Persisted output workbook does not contain exactly 100 rows."
        )

    # the four input columns should come back unchanged
    for column in EXPECTED_COLUMNS:
        if not persisted_df[column].equals(df[column]):
            fail(
                f"Frozen input column was changed in output: {column}"
            )

    # and every reply should read back exactly as it was written
    for index in range(100):
        original_reply = generated_replies[index]
        persisted_reply = persisted_df.loc[index, "Generated Reply"]

        if persisted_reply != original_reply:
            fail(
                "Generated Reply changed after writing/reopening the "
                f"workbook at output row {index + 1}."
            )

    write_metadata(
        successful_generations=successful_generations,
        failed_generations=failed_generations,
    )

    # summary
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Number of input rows: {len(df)}")
    print(
        "Condition A generations: "
        f"{(df['Condition'] == 'A').sum()}"
    )
    print(
        "Condition B generations: "
        f"{(df['Condition'] == 'B').sum()}"
    )
    print("Total generations: 100")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Total generation time: {total_generation_time:.3f} sec")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Metadata file: {METADATA_FILE}")
    print("Generation status = PASSED")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print("Generation interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print()
        print("=" * 70)
        print("Generation status = FAILED")
        print("=" * 70)
        print(f"Error: {exc}")
        sys.exit(1)
