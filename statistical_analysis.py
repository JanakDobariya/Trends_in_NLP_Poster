from pathlib import Path
import math

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "evaluation" / "human_rated_results.xlsx"
OUTPUT_DIR = BASE_DIR / "analysis"

STATISTICAL_SEED = 2026
ALPHA = 0.05

EXPECTED_COLUMNS = [
    "Sr.No.",
    "Conversation ID",
    "Condition",
    "Context",
    "Generated Reply",
    "Coherence",
    "Relevance",
    "Plausibility",
    "Conversational Appropriateness",
    "Contextual Fit",
]

DIMENSIONS = [
    "Coherence",
    "Relevance",
    "Plausibility",
    "Conversational Appropriateness",
    "Contextual Fit",
]

OUTPUT_DIMENSION_COLUMNS = {
    "Coherence": "coherence",
    "Relevance": "relevance",
    "Plausibility": "plausibility",
    "Conversational Appropriateness": "conversational_appropriateness",
    "Contextual Fit": "contextual_fit",
}

OVERALL_LABEL = "Overall Score"

def fail(message):
    raise RuntimeError(message)


def format_number(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def format_p(value):
    if value is None or not np.isfinite(value):
        return "NA"
    if value < 0.0001:
        return f"{value:.4g}"
    return f"{value:.4f}"


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        return float("nan"), float("nan")

    mean_value = np.mean(values)
    sd = np.std(values, ddof=1)
    se = sd / math.sqrt(n)
    critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    margin = critical * se
    return mean_value - margin, mean_value + margin


def paired_mean_ci(differences, confidence=0.95):
    """95% t-interval for the mean of the paired differences (B - A)."""
    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    if n < 2:
        return float("nan"), float("nan")

    mean_difference = np.mean(differences)
    sd_difference = np.std(differences, ddof=1)
    se = sd_difference / math.sqrt(n)

    critical = stats.t.ppf(
        (1 + confidence) / 2,
        df=n - 1,
    )

    margin = critical * se
    return mean_difference - margin, mean_difference + margin


# Loading and checking the ratings

def load_and_validate_input():
    """Read the rating workbook and stop if anything in its structure is off."""

    if not INPUT_FILE.exists():
        fail(f"Input file does not exist: {INPUT_FILE}")

    try:
        df = pd.read_excel(INPUT_FILE)
    except Exception as exc:
        fail(f"Could not read input workbook: {exc}")

    if len(df) != 100:
        fail(
            f"Expected exactly 100 rated responses, "
            f"but found {len(df)}."
        )

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        fail(
            "Input workbook columns do not match the expected schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Found:    {actual_columns}"
        )

    if df["Conversation ID"].isna().any():
        fail("Conversation ID contains missing values.")

    if df["Condition"].isna().any():
        fail("Condition contains missing values.")

    conversation_counts = df["Conversation ID"].value_counts()

    if len(conversation_counts) != 50:
        fail(
            f"Expected exactly 50 unique Conversation IDs, "
            f"found {len(conversation_counts)}."
        )

    if not (conversation_counts == 2).all():
        bad_ids = conversation_counts[conversation_counts != 2]
        fail(
            "Every Conversation ID must occur exactly twice.\n"
            f"Invalid IDs:\n{bad_ids.to_string()}"
        )

    condition_counts = df["Condition"].value_counts()

    if condition_counts.get("A", 0) != 50:
        fail(
            f"Expected exactly 50 Condition A rows, "
            f"found {condition_counts.get('A', 0)}."
        )

    if condition_counts.get("B", 0) != 50:
        fail(
            f"Expected exactly 50 Condition B rows, "
            f"found {condition_counts.get('B', 0)}."
        )

    unexpected_conditions = set(condition_counts.index) - {"A", "B"}

    if unexpected_conditions:
        fail(
            f"Unexpected condition values found: "
            f"{sorted(unexpected_conditions)}"
        )

    # each conversation needs one A row and one B row
    for conversation_id, group in df.groupby("Conversation ID", sort=False):
        conditions = list(group["Condition"])

        if conditions.count("A") != 1 or conditions.count("B") != 1:
            fail(
                f"Conversation {conversation_id!r} does not contain "
                "exactly one A and one B response."
            )

    # ratings: no missing values, numeric, between 1 and 5
    for dimension in DIMENSIONS:
        if dimension not in df.columns:
            fail(f"Missing rating column: {dimension}")

        if df[dimension].isna().any():
            missing_rows = df.index[df[dimension].isna()].tolist()
            fail(
                f"Rating column {dimension!r} contains missing values "
                f"at rows: {missing_rows}"
            )

        numeric_values = pd.to_numeric(df[dimension], errors="coerce")

        if numeric_values.isna().any():
            bad_rows = df.index[numeric_values.isna()].tolist()
            fail(
                f"Rating column {dimension!r} contains non-numeric "
                f"values at rows: {bad_rows}"
            )

        if ((numeric_values < 1) | (numeric_values > 5)).any():
            bad_rows = df.index[
                (numeric_values < 1) | (numeric_values > 5)
            ].tolist()

            fail(
                f"Rating column {dimension!r} contains values outside "
                f"the permitted 1–5 range at rows: {bad_rows}"
            )

        # use floats for the calculations
        df[dimension] = numeric_values.astype(float)

    if df["Context"].isna().any():
        fail("Context contains missing values.")

    if (df["Context"].astype(str).str.strip() == "").any():
        fail("Context contains empty values.")

    if df["Generated Reply"].isna().any():
        fail("Generated Reply contains missing values.")

    if (df["Generated Reply"].astype(str).str.strip() == "").any():
        fail("Generated Reply contains empty values.")

    return df


# Building the A/B pairs

def construct_pairs(df):
# One row per conversation with the A ratings, the B ratings and B - A (overall score included).

    pair_records = []

    for conversation_id, group in df.groupby(
        "Conversation ID",
        sort=True,
    ):
        a_rows = group[group["Condition"] == "A"]
        b_rows = group[group["Condition"] == "B"]

        if len(a_rows) != 1 or len(b_rows) != 1:
            fail(
                f"Conversation {conversation_id!r} cannot be paired: "
                f"A={len(a_rows)}, B={len(b_rows)}."
            )

        a = a_rows.iloc[0]
        b = b_rows.iloc[0]

        record = {
            "Conversation ID": conversation_id,
        }

        for dimension in DIMENSIONS:
            short_name = OUTPUT_DIMENSION_COLUMNS[dimension]

            a_value = float(a[dimension])
            b_value = float(b[dimension])

            record[f"A_{short_name}"] = a_value
            record[f"B_{short_name}"] = b_value
            record[f"B_minus_A_{short_name}"] = b_value - a_value

        a_overall = np.mean([float(a[d]) for d in DIMENSIONS])
        b_overall = np.mean([float(b[d]) for d in DIMENSIONS])

        record["A_Overall"] = a_overall
        record["B_Overall"] = b_overall
        record["B_minus_A_Overall"] = b_overall - a_overall

        pair_records.append(record)

    pairs = pd.DataFrame(pair_records)

    if len(pairs) != 50:
        fail(
            f"Expected 50 complete A/B pairs, "
            f"but constructed {len(pairs)}."
        )

    return pairs


# Statistics for one measure

def analyze_measure(pairs, measure_name, short_name):
    #Descriptives, 95% CI, Cohen's dz and the Wilcoxon test for one measure.

    a = pairs[f"A_{short_name}"].to_numpy(dtype=float)
    b = pairs[f"B_{short_name}"].to_numpy(dtype=float)
    differences = b - a

    n = len(differences)

    # descriptives
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    a_sd = np.std(a, ddof=1)
    b_sd = np.std(b, ddof=1)

    a_median = np.median(a)
    b_median = np.median(b)

    a_min = np.min(a)
    a_max = np.max(a)

    b_min = np.min(b)
    b_max = np.max(b)

    difference_mean = np.mean(differences)
    difference_sd = np.std(differences, ddof=1)
    difference_median = np.median(differences)
    difference_min = np.min(differences)
    difference_max = np.max(differences)

    ci_low, ci_high = paired_mean_ci(differences)

    if difference_sd == 0:
        cohen_dz = 0.0 if difference_mean == 0 else np.sign(difference_mean) * np.inf
    else:
        cohen_dz = difference_mean / difference_sd

    # two-sided Wilcoxon signed-rank test

    nonzero_count = int(np.count_nonzero(differences))

    if nonzero_count == 0:
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0
    else:
        wilcoxon_result = stats.wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)

    return {
        "measure": measure_name,
        "n_pairs": n,
        "A_mean": a_mean,
        "A_SD": a_sd,
        "A_median": a_median,
        "A_min": a_min,
        "A_max": a_max,
        "B_mean": b_mean,
        "B_SD": b_sd,
        "B_median": b_median,
        "B_min": b_min,
        "B_max": b_max,
        "B_minus_A_mean": difference_mean,
        "B_minus_A_SD": difference_sd,
        "B_minus_A_median": difference_median,
        "B_minus_A_min": difference_min,
        "B_minus_A_max": difference_max,
        "CI95_low": ci_low,
        "CI95_high": ci_high,
        "Cohen_dz": cohen_dz,
        "Wilcoxon_statistic": wilcoxon_statistic,
        "Wilcoxon_p": wilcoxon_p,
        "Holm_adjusted_p": np.nan,
        "primary_test": "Two-sided Wilcoxon signed-rank",
        "analysis_role": "PRIMARY",
    }


# Holm-Bonferroni correction

def holm_bonferroni(p_values):

# Holm-adjusted p-values. Sort ascending

    p_values = np.asarray(p_values, dtype=float)

    m = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, index in enumerate(order):
        adjusted_value = (m - rank) * p_values[index]
        running_max = max(running_max, adjusted_value)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def apply_holm_correction(results_df):
# Adjust the five primary p-values

    primary_mask = results_df["analysis_role"] == "PRIMARY"

    primary_p = results_df.loc[
        primary_mask,
        "Wilcoxon_p",
    ].to_numpy(dtype=float)

    if len(primary_p) != 5:
        fail(
            "Expected exactly five primary Wilcoxon p-values "
            f"for Holm correction, found {len(primary_p)}."
        )

    adjusted = holm_bonferroni(primary_p)

    results_df.loc[
        primary_mask,
        "Holm_adjusted_p",
    ] = adjusted

    # the overall score keeps Holm_adjusted_p = NaN
    return results_df

def create_paired_differences_file(pairs):

    columns = ["Conversation ID"]

    for dimension in DIMENSIONS:
        short_name = OUTPUT_DIMENSION_COLUMNS[dimension]

        columns.extend([
            f"A_{short_name}",
            f"B_{short_name}",
            f"B_minus_A_{short_name}",
        ])

    columns.extend([
        "A_Overall",
        "B_Overall",
        "B_minus_A_Overall",
    ])

    output = pairs[columns].copy()

    output_path = OUTPUT_DIR / "paired_differences.csv"
    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path

def create_results_file(results_df):

    columns = [
        "measure",
        "n_pairs",
        "A_mean",
        "B_mean",
        "B_minus_A_mean",
        "CI95_low",
        "CI95_high",
        "Wilcoxon_statistic",
        "Wilcoxon_p",
        "Holm_adjusted_p",
        "Cohen_dz",
        "analysis_role",
    ]

    output = results_df[columns].copy()

    output_path = OUTPUT_DIR / "statistical_results.csv"

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path

def create_dimension_means_figure(results_df):

    primary = results_df[
        results_df["analysis_role"] == "PRIMARY"
    ].copy()

    labels = primary["measure"].tolist()
    a_means = primary["A_mean"].to_numpy(dtype=float)
    b_means = primary["B_mean"].to_numpy(dtype=float)

    a_errors = []
    b_errors = []

    for _, row in primary.iterrows():
        n = int(row["n_pairs"])

        a_se = row["A_SD"] / math.sqrt(n)
        b_se = row["B_SD"] / math.sqrt(n)

        critical = stats.t.ppf(
            0.975,
            df=n - 1,
        )

        a_errors.append(critical * a_se)
        b_errors.append(critical * b_se)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - width / 2,
        a_means,
        width,
        yerr=a_errors,
        capsize=4,
        label="Condition A: Parent only",
    )

    ax.bar(
        x + width / 2,
        b_means,
        width,
        yerr=b_errors,
        capsize=4,
        label="Condition B: Grandparent + Parent",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean human rating (1–5)")
    ax.set_xlabel("Rating dimension")
    ax.set_ylim(1, 5.2)
    ax.set_title("Condition A vs Condition B: Mean Human Ratings")
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIR / "dimension_means.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def create_overall_score_figure(results_df):
   # A vs B mean overall score with 95% error bars

    row = results_df[
        results_df["measure"] == OVERALL_LABEL
    ].iloc[0]

    n = int(row["n_pairs"])

    critical = stats.t.ppf(
        0.975,
        df=n - 1,
    )

    a_error = critical * row["A_SD"] / math.sqrt(n)
    b_error = critical * row["B_SD"] / math.sqrt(n)

    fig, ax = plt.subplots(figsize=(7, 6))

    x = np.array([0, 1])

    ax.errorbar(
        x,
        [row["A_mean"], row["B_mean"]],
        yerr=[a_error, b_error],
        fmt="o",
        capsize=5,
        markersize=8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([
        "Condition A\nParent only",
        "Condition B\nGrandparent + Parent",
    ])

    ax.set_ylabel("Mean overall score (1–5)")
    ax.set_ylim(1, 5.2)
    ax.set_title("Overall Human-Rated Quality")

    fig.tight_layout()

    path = OUTPUT_DIR / "overall_score.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def create_paired_difference_figure(results_df):
    # Mean B - A difference with its 95% CI for every measure

    plot_df = results_df.copy()

    labels = plot_df["measure"].tolist()
    means = plot_df["B_minus_A_mean"].to_numpy(dtype=float)

    ci_low = plot_df["CI95_low"].to_numpy(dtype=float)
    ci_high = plot_df["CI95_high"].to_numpy(dtype=float)

    lower_errors = means - ci_low
    upper_errors = ci_high - means

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        x,
        means,
        yerr=[lower_errors, upper_errors],
        fmt="o",
        capsize=5,
        markersize=7,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean paired difference (B − A)")
    ax.set_xlabel("Measure")
    ax.set_title(
        "Paired Difference: Condition B − Condition A"
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "paired_differences.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


# Checks on the results
def validate_output_results(results_df, pairs):
    # Sanity checks on the results before anything is written

    if len(results_df) != 6:
        fail(
            f"Expected 6 statistical result rows, "
            f"found {len(results_df)}."
        )

    expected_measures = DIMENSIONS + [OVERALL_LABEL]

    if results_df["measure"].tolist() != expected_measures:
        fail(
            "Statistical result measures are incorrect.\n"
            f"Expected: {expected_measures}\n"
            f"Found: {results_df['measure'].tolist()}"
        )

    if len(pairs) != 50:
        fail(
            f"Expected exactly 50 paired conversations, found {len(pairs)}."
        )

    if pairs["Conversation ID"].nunique() != 50:
        fail("Paired differences do not contain exactly 50 unique conversations.")

    primary = results_df[
        results_df["analysis_role"] == "PRIMARY"
    ]

    secondary = results_df[
        results_df["analysis_role"] == "SECONDARY"
    ]

    if len(primary) != 5:
        fail(
            f"Expected 5 PRIMARY measures, found {len(primary)}."
        )

    if len(secondary) != 1:
        fail(
            f"Expected 1 SECONDARY measure, found {len(secondary)}."
        )

    if secondary.iloc[0]["measure"] != OVERALL_LABEL:
        fail("Overall Score must be the sole SECONDARY measure.")

    if primary["Holm_adjusted_p"].isna().any():
        fail("A primary Holm-adjusted p-value is missing.")

    # the overall score should not have a Holm value
    if not np.isnan(
        secondary.iloc[0]["Holm_adjusted_p"]
    ):
        fail(
            "Overall Score must not receive a Holm-adjusted p-value "
            "from the primary five-test family."
        )


def main():
    print("=" * 50)
    print("FINAL STATISTICAL ANALYSIS")
    print("=" * 50)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input file: {INPUT_FILE}")

    df = load_and_validate_input()

    print(f"Number of rated responses: {len(df)}")
    print(
        f"Number of unique conversations: "
        f"{df['Conversation ID'].nunique()}"
    )

    pairs = construct_pairs(df)

    print(
        f"Number of complete A/B pairs: {len(pairs)}"
    )

    # the five primary dimensions

    results = []

    for dimension in DIMENSIONS:
        short_name = OUTPUT_DIMENSION_COLUMNS[dimension]

        result = analyze_measure(pairs, dimension, short_name)

        results.append(result)

    # overall score (secondary outcome)

    overall_result = analyze_measure(
        pairs,
        OVERALL_LABEL,
        "Overall",
    )

    overall_result["analysis_role"] = "SECONDARY"

    results.append(overall_result)

    results_df = pd.DataFrame(results)

    # Holm correction over the five primary dimensions

    results_df = apply_holm_correction(results_df)

    validate_output_results(results_df, pairs)

    # write the CSV files and figures

    results_path = create_results_file(results_df)

    paired_path = create_paired_differences_file(pairs)

    dimension_figure = create_dimension_means_figure(results_df)

    overall_figure = create_overall_score_figure(results_df)

    paired_figure = create_paired_difference_figure(results_df)


    print("")
    print("=" * 50)
    print("FINAL STATISTICAL ANALYSIS")
    print("=" * 50)

    print(f"Input file: {INPUT_FILE}")
    print(f"Number of rated responses: {len(df)}")
    print(
        f"Number of unique conversations: "
        f"{df['Conversation ID'].nunique()}"
    )
    print(f"Number of complete A/B pairs: {len(pairs)}")
    print("")

    print("Primary dimensions:")
    for dimension in DIMENSIONS:
        print(f"- {dimension}")

    print("")
    print("Primary test:")
    print("Two-sided Wilcoxon signed-rank")

    print("")
    print("Multiple-comparison correction:")
    print("Holm-Bonferroni")

    print("")
    print("Overall score:")
    print("Secondary analysis")

    print("")
    print(f"Output directory: {OUTPUT_DIR}")
    print("")
    print("Created files:")
    print(f"- {results_path}")
    print(f"- {paired_path}")
    print(f"- {dimension_figure}")
    print(f"- {overall_figure}")
    print(f"- {paired_figure}")
    print("")
    print("Analysis status = PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
