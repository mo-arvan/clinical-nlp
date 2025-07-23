import argparse
import csv
import functools
import hashlib
import json
import logging
import random
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import litellm
from litellm import RateLimitError  # Import the RateLimitError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random
from tqdm import tqdm
import medspacy
import numpy as np
import pandas as pd
import spacy
from medspacy.ner import TargetRule
from pydantic import BaseModel

import parallel_helper
import target_matcher
from rulebook import cervical_rulebook
from pipeline import PipelineConfig, StructuredLLMPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Running {__file__}")

target_matcher.init()


class AssertionCategory(str, Enum):
    POSITIVE = "positive"
    NEGATED = "negated"
    POSSIBLE = "possible"
    HYPOTHETICAL = "hypothetical"
    FAMILY = "family"
    HISTORICAL = "historical"


class ConceptExtractionVerification(BaseModel):
    reasoning: str
    assertion: AssertionCategory


def load_cervical_rulebook():
    rule_set = []
    for rule in cervical_rulebook.cervical_rulebook_definition:
        metadata = rule.get("metadata", None)
        if "pattern" in rule:
            rule_set.append(
                TargetRule(
                    literal=None,
                    category=rule["category"],
                    pattern=rule["pattern"],
                    metadata=metadata,
                )
            )
        elif "literal" in rule:
            rule_set.append(
                TargetRule(
                    literal=rule["literal"],
                    category=rule["category"],
                    pattern=None,
                    metadata=metadata,
                )
            )
        else:
            logger.error(f"Invalid rule {rule}")

    # rule_set = [rule_set[1]]
    return rule_set


def load_cervical_data():
    data_path = "datasets/cervical/data_with_phi/holt_2023_00185_pap_hpv_deid.csv"
    cervical_validation_set = "datasets/cervical/cervical-validation.json"
    test_1_set = "datasets/cervical/cervical-test-1-20240827.json"
    test_2_set = "datasets/cervical/cervical-test-2-20241029.json"
    notes_column = "PROC_NARRATIVE_CLEAN"  # "Proc narrative"

    notes_df = pd.read_csv(data_path)
    duplicate_notes_condition = notes_df.duplicated(keep=False)
    notes_df[duplicate_notes_condition].to_csv(
        "artifacts/results/cervical_duplicate_notes.csv", index=False
    )

    notes_df["note_id"] = range(len(notes_df))
    notes_df["note_hash"] = [
        hashlib.md5(note.encode()).hexdigest() for note in notes_df["PROC_NARRATIVE"]
    ]
    # notes_df = notes_df[~duplicate_notes_condition]
    notes_df = notes_df.drop_duplicates()
    notes_df.fillna(value="Not Available", inplace=True)

    notes_df["PROC_NARRATIVE_CLEAN"] = notes_df["PROC_NARRATIVE"].apply(
        lambda x: x.replace("\\n", "\n")
    )

    with open(cervical_validation_set, "r") as f:
        validation_labels = json.load(f)

    with open(test_1_set, "r") as f:
        test_1_set = json.load(f)

    with open(test_2_set, "r") as f:
        test_2_set = json.load(f)

    validation_set_ids = [note["data"]["note_id"] for note in validation_labels]
    test_1_set_ids = [note["data"]["note_id"] for note in test_1_set]
    test_2_set_ids = [note["data"]["note_id"] for note in test_2_set]
    valid_df = notes_df[notes_df["note_id"].isin(validation_set_ids)]
    test_1_df = notes_df[notes_df["note_id"].isin(test_1_set_ids)]
    test_2_df = notes_df[notes_df["note_id"].isin(test_2_set_ids)]

    valid_test_ids = validation_set_ids + test_1_set_ids + test_2_set_ids

    rest_df = notes_df[~notes_df["note_id"].isin(valid_test_ids)]

    def get_labels_df(cervical_labels):
        gold_labels = [
            (
                l["data"]["note_id"],
                rr["value"]["start"],
                rr["value"]["end"],
                rr["value"]["text"],
                rrr,
            )
            for l in cervical_labels
            for r in l["annotations"]
            for rr in r["result"]
            for rrr in rr["value"]["labels"]
        ]
        labels_df = pd.DataFrame(
            gold_labels, columns=["file_id", "start", "end", "text", "label"]
        )
        # duplicate_labels_condition = labels_df.duplicated(keep=False)
        # duplicate_labels = labels_df[duplicate_labels_condition]
        # duplicates_removed = duplicate_labels.drop_duplicates()
        # labels_df = labels_df[~duplicate_labels_condition]
        labels_df = labels_df.drop_duplicates()
        return labels_df

    valid_labels_df = get_labels_df(validation_labels)
    test_1_labels = get_labels_df(test_1_set)
    test_2_labels = get_labels_df(test_2_set)

    annotation_count = len(valid_labels_df) + len(test_1_labels) + len(test_2_labels)
    logger.info(f"Notes annotated: {len(valid_test_ids)}")
    logger.info(f"Total annotations: {annotation_count}")

    dataset_dict = {
        "valid": {"notes": valid_df, "labels": valid_labels_df},
        "test-1": {"notes": test_1_df, "labels": test_1_labels},
        "test-2": {"notes": test_2_df, "labels": test_2_labels},
        "train": {"notes": rest_df, "labels": []},
    }

    return dataset_dict, notes_column


def load_cervical_data_latest():
    data_dir = "datasets/cervical/data_with_phi/2024_12_18_Patient data"
    data_file = "datasets/cervical/data_with_phi/2024_12_18_Patient data/req_2023_00185_procs_hpv.csv"
    notes_column = (
        "RESULT"  # Using RESULT column as it contains the procedure narrative
    )

    # Read the CSV data
    notes_df = pd.read_csv(data_file, delimiter="|")

    # Handle duplicates
    duplicate_notes_condition = notes_df.duplicated(keep=False)
    notes_df[duplicate_notes_condition].to_csv(
        "artifacts/results/cervical_duplicate_notes.csv", index=False
    )

    # Add note IDs and hash values
    notes_df["note_id"] = range(len(notes_df))
    notes_df["note_hash"] = [
        hashlib.md5(note.encode()).hexdigest() for note in notes_df[notes_column]
    ]
    notes_df = notes_df.drop_duplicates()

    # Fill missing values
    notes_df.fillna(value="Not Available", inplace=True)

    # Clean newlines in text
    notes_df[f"{notes_column}_CLEAN"] = notes_df[notes_column].apply(
        lambda x: x.replace("\\n", "\n")
    )

    # split the notes into two groups, the ones that have "history" in the text and the ones that don't

    valid_df = notes_df[
        notes_df[notes_column].str.contains("history", case=False, na=False)
    ]
    train_df = notes_df[
        ~notes_df[notes_column].str.contains("history", case=False, na=False)
    ]

    # Log dataset statistics
    logger.info(f"Total notes: {len(notes_df)}")

    # Create the dataset dictionary - all data goes to train since we don't have labels
    dataset_dict = {
        "train": {
            "notes": train_df,
            "labels": pd.DataFrame(),  # No labels available
        },
        "valid": {
            "notes": valid_df,
            "labels": pd.DataFrame(),  # No labels available
        },
    }

    return (
        dataset_dict,
        f"{notes_column}_CLEAN",
    )  # Return the clean column name for processing


def get_medspacy_label(ent):
    if not hasattr(ent._, "modifiers"):
        return "present"

    context_to_i2b2_label_map = {
        "NEGATED_EXISTENCE": "absent",
        "POSSIBLE_EXISTENCE": "possible",
        "CONDITIONAL_EXISTENCE": "conditional",
        "HYPOTHETICAL": "hypothetical",
        "HISTORICAL": "historical",
        "FAMILY": "family",
    }

    modifiers_category = [mod.category for mod in ent._.modifiers]

    label = None
    if len(modifiers_category) == 0:
        # no modifiers, assume present
        label = "present"
    elif len(modifiers_category) == 1:
        # we have a single modifier, we report it
        label = context_to_i2b2_label_map[modifiers_category[0]]
    else:
        # more than one modifier, we report the most frequent one
        # we decide an order of precedence
        # 1. absent
        # 2. possible
        # 3. hypothetical
        # 4. conditional
        # 5. not associated
        if ent._.is_uncertain:
            label = "possible"
        elif ent._.is_negated:
            label = "absent"
        # currently we cannot handle conditional
        # elif ent._.is_conditional:
        #     label = "conditional"
        elif ent._.is_hypothetical:
            label = "hypothetical"
        # i2b2 does not have historical labels
        # elif ent._.is_historical:
        #     label = "historical"
        elif ent._.is_family:
            label = "family"

    return label


def run_medspacy_on_note(i_data_row, medspacy_nlp, note_text_column):
    i, data_row = i_data_row

    doc = medspacy_nlp(data_row[note_text_column])

    # if i == 234:
    #     print("Debug")

    row_dict = data_row.to_dict()
    note_id = data_row["note_id"]

    prediction_list = []
    for ent in doc.ents:
        text = ent.text
        label = ent.label_
        assertion = get_medspacy_label(ent)
        prediction_id = f"note_{note_id}_ent_{ent.start}_{ent.end}"

        prediction_dict = {
            "value": {
                "start": ent.start_char,
                "end": ent.end_char,
                "text": text,
                "labels": [label, assertion],
            },
            "id": prediction_id,
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "prediction",
        }
        prediction_list.append(prediction_dict)

    results_dict = {
        "id": note_id,
        "data": {"num_predictions": len(prediction_list), **row_dict},
        "predictions": [
            {
                "result": prediction_list,
            }
        ],
    }

    return results_dict


def run_medspacy_prediction_pipeline(notes_df, rule_set, notes_column):
    nlp = medspacy.load(
        # medspacy_enable=[
        # "medspacy_pyrush",
        # "medspacy_target_matcher",
        # "medspacy_context"
        # ]
    )

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    for rule in rule_set:
        try:
            target_matcher.add(rule)
        except Exception as e:
            print(f"Failed to add rule {rule}")
            print(e)
    target_matcher._TargetMatcher__matcher._prune = False

    run_medspacy_on_note_partial = functools.partial(
        run_medspacy_on_note, medspacy_nlp=nlp, note_text_column=notes_column
    )
    doc_results_list = parallel_helper.run_in_parallel_cpu_bound(
        run_medspacy_on_note_partial,
        notes_df.iterrows(),
        total=len(notes_df),
        max_workers=16,
    )

    return doc_results_list


def run_prediction_pipeline(notes_df, rule_set, notes_column):
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.blank("en")

    nlp.add_pipe("target_matcher")
    target_matcher = nlp.get_pipe("target_matcher")
    target_matcher.add(rule_set)

    # debug_note_list = [6475]  # 349 6723, 6724 6929
    # notes_df = notes_df[notes_df["note_id"].isin(debug_note_list)]

    run_medspacy_on_note_partial = functools.partial(
        run_medspacy_on_note, medspacy_nlp=nlp, note_text_column=notes_column
    )
    doc_results_list = parallel_helper.run_in_parallel_cpu_bound(
        run_medspacy_on_note_partial,
        notes_df.iterrows(),
        total=len(notes_df),
        max_workers=16,
    )

    return doc_results_list


def export_results(doc_results_list, labels_list, out_dir):
    result_list = [result for doc in doc_results_list for result in doc if len(doc) > 0]

    results_df = pd.DataFrame(result_list)

    results_df.to_csv("artifacts/results/cervical_notes.csv", index=False)

    match_pattern_counts = (
        results_df[["matched_pattern"]]
        .groupby("matched_pattern")
        .size()
        .reset_index(name="count")
    )

    missing_patterns = []
    for label, pattern in labels_list:
        if pattern not in match_pattern_counts["matched_pattern"].values:
            missing_patterns.append((pattern, 0))

    missing_patterns_df = pd.DataFrame(
        missing_patterns, columns=["matched_pattern", "count"]
    )
    match_pattern_counts = pd.concat([match_pattern_counts, missing_patterns_df])

    match_pattern_counts = match_pattern_counts.sort_values(by="count", ascending=False)

    match_pattern_counts.to_csv(
        "artifacts/results/cervical_notes_match_pattern_counts.csv", index=False
    )

    return results_df


def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def sample_results(test_results, valid_results, k):
    selected_results = []
    selected_results_label_count_dict = {}
    text_to_notes_dict = {}
    label_to_notes_dict = {}
    id_to_labels_dict = {}
    for i, doc in enumerate(test_results):
        for prediction in doc["predictions"]:
            result = prediction["result"]

            for r in result:
                text = r["value"]["text"]
                for label in r["value"]["labels"]:
                    if label in [
                        "present",
                        "absent",
                        "possible",
                        "conditional",
                        "hypothetical",
                        "historical",
                        "family",
                    ]:
                        continue
                    if label not in label_to_notes_dict:
                        label_to_notes_dict[label] = []
                    label_to_notes_dict[label].append(i)

                    if i not in id_to_labels_dict:
                        id_to_labels_dict[i] = []
                    id_to_labels_dict[i].append(label)

                if text not in text_to_notes_dict:
                    text_to_notes_dict[text] = []
                text_to_notes_dict[text].append(i)

    text_to_notes_dict = dict(
        sorted(text_to_notes_dict.items(), key=lambda x: len(x[1]))
    )
    label_to_notes_dict = dict(
        sorted(label_to_notes_dict.items(), key=lambda x: len(x[1]))
    )
    label_to_support = dict(zip(valid_results["label"], valid_results["support"]))
    label_to_support = dict(sorted(label_to_support.items(), key=lambda x: x[1]))

    max_per_label = 15

    for label in label_to_support.keys():
        if label in label_to_notes_dict:
            l = min(max_per_label, len(label_to_notes_dict[label]))

            selected_results.extend(random.sample(label_to_notes_dict[label], l))
            if label not in selected_results_label_count_dict:
                selected_results_label_count_dict[label] = 0
            selected_results_label_count_dict[label] += l

    selected_results = list(set(selected_results))

    # selected_notes = []
    # for text, notes in text_to_notes_dict.items():
    #     for note_index in notes:
    #         if note_index not in selected_notes:
    #             selected_notes.append(note_index)
    sampled_results = [test_results[i] for i in selected_results[:k]]
    return sampled_results


def export_as_label_studio_format(
    doc_results_list, out_dir, file_name, remove_predictions=False
):
    out_file_path = f"{out_dir}/{file_name}"

    results_list = [doc for doc in doc_results_list if len(doc) > 0]

    if remove_predictions:
        for i, doc in enumerate(results_list):
            results_list[i]["predictions"] = []
    with open(out_file_path, "w") as f:
        json.dump(results_list, f, indent=2)


def misc_func():
    export_as_label_studio_format(doc_results_list, "artifacts/results/")

    from collections import Counter

    l = Counter(
        [
            r["value"]["text"]
            for p in doc_results_list
            for r in p["predictions"][0]["result"]
        ]
    )
    l = sorted(l.items(), key=lambda x: x[1], reverse=True)
    print(l)
    labels = set(
        [
            ll
            for p in doc_results_list
            for r in p["predictions"][0]["result"]
            for ll in r["value"]["labels"]
        ]
    )

    known_labels = [r.category for r in rule_set]

    missing_labels = [l for l in known_labels if l not in labels]
    print(missing_labels)


def evaluate2():
    results_df.rename(columns={"label": "pred_label"}, inplace=True)

    results_df_joined = results_df.merge(
        cervical_labels,
        on=["prediction_id", "matched_pattern", "pred_label"],
        how="left",
    )

    eval_df = (
        results_df_joined[["pred_label", "gold_label"]]
        .groupby(["pred_label", "gold_label"])
        .size()
        .reset_index(name="count")
    )

    micro_average = (
        eval_df.groupby("gold_label").agg(count=("count", "sum")).reset_index()
    )
    annotation_values = ["false positive", "false negative", "true positive"]

    for annotation in annotation_values:
        if annotation not in micro_average["gold_label"].values:
            micro_average.loc[len(micro_average)] = [annotation, 0]

    micro_average["pred_label"] = "micro_average"

    # concatenate the two dataframes
    eval_df = pd.concat([eval_df, micro_average], ignore_index=True)
    confusion_matrix_df = eval_df.pivot(
        index="pred_label", columns="gold_label", values="count"
    ).fillna(0)

    confusion_matrix_df.to_csv("artifacts/results/cervical_confusion_matrix.csv")
    results_df_joined.to_csv(
        "artifacts/results/cervical_notes_with_labels.csv", index=False
    )
    results_df_joined.to_excel(
        "artifacts/results/cervical_notes_with_labels.xlsx", index=False
    )
    results_labeled = results_df_joined.dropna(subset=["gold_label"])

    labels_count = (
        results_labeled[["matched_pattern", "pred_label"]]
        .groupby(["matched_pattern", "pred_label"])
        .size()
        .reset_index(name="count")
    )

    labels_count.to_csv("artifacts/results/cervical_labels_count.csv", index=False)


# Function to determine if two spans intersect
def do_intersect(start1, end1, start2, end2):
    return max(start1, start2) <= min(end1, end2)


# Function to find matches according to the new criteria
def classify_matches(labels_df, predictions_df):
    labels_list = labels_df.to_dict("records")
    predictions_list = predictions_df.to_dict("records")
    tp, fp, fn = 0, 0, 0
    classification_report = []
    # Check each prediction
    for pred_dict in predictions_list:
        match_found = False
        for index, true_row in enumerate(labels_list):
            if (
                pred_dict["file_id"] == true_row["file_id"]
                and pred_dict["label_pred"] == true_row["label_gold"]
                and do_intersect(
                    pred_dict["start"],
                    pred_dict["end"],
                    true_row["start"],
                    true_row["end"],
                )
            ):
                match_found = True
                labels_list[index]["matched"] = True
                tp += 1
                classification_report.append(
                    {
                        "file_id": pred_dict["file_id"],
                        "label": pred_dict["label_pred"],
                        "classification": "true_positive",
                        "pred_start": pred_dict["start"],
                        "pred_end": pred_dict["end"],
                        "gold_start": true_row["start"],
                        "gold_end": true_row["end"],
                        "pred_text": pred_dict["text_pred"],
                        "gold_text": true_row["text_gold"],
                    }
                )
                break
        if not match_found:
            fp += 1
            classification_report.append(
                {
                    "file_id": pred_dict["file_id"],
                    "label": pred_dict["label_pred"],
                    "classification": "false_positive",
                    "pred_start": pred_dict["start"],
                    "pred_end": pred_dict["end"],
                    "pred_text": pred_dict["text_pred"],
                }
            )
    # Remaining entries are false negatives
    remaining_labels = [
        true_row for true_row in labels_list if "matched" not in true_row
    ]
    fn = len(remaining_labels)
    for fn_dict in remaining_labels:
        classification_report.append(
            {
                "file_id": fn_dict["file_id"],
                "label": fn_dict["label_gold"],
                "classification": "false_negative",
                "gold_start": fn_dict["start"],
                "gold_end": fn_dict["end"],
                "gold_text": fn_dict["text_gold"],
            }
        )

    return tp, fp, fn, classification_report


def evaluate_and_report(
    doc_results_list, labels_df, eval_set="test-1", filter_labels=True
):
    desired_columns = [
        "Negative/Normal/NILM",
        "Atypical Squamous Cells of undetermined Significance",
        "Atypical squamous cells cannot exclude HSIL atypical squamous cells (ASC-H)",
        "Low-grade squamous intraepithelial lesion (LSIL)",
        "High-grade squamous intraepithelial lesion (HSIL)",
        "High-grade squamous intraepithelial lesion (HGSIL)",
        "Squamous Cell Carcinoma",
        "Atypical endocervical cells",
        "Atypical endometrial cells",
        "Atypical glandular cells",
        "Atypical glandular cells (favors neoplastic)",
        "Endocervical Adenocarcinoma in Situ (AIS)",
        "Adenocarcinoma",
        "Endocervical adenocarcinoma",
        "Endometrial adenocarcinoma",
        "Extrauterine adenocarcinoma",
        "Adenocarcinoma NOS",
        "HPV Negative",
        "HPV Positive",
        "HPV 16 Negative",
        "HPV 16 Positive",
        "HPV 18 Negative",
        "HPV 18 Positive",
        "HPV 18/45 Negative",
        "HPV 18/45 positive",
        "HPV Other Negative",
        "HPV Other positive",
    ]

    prediction_list = [
        (doc["id"], r["value"]["start"], r["value"]["end"], r["value"]["text"], rr)
        for doc in doc_results_list
        for prediction in doc["predictions"]
        for r in prediction["result"]
        for rr in r["value"]["labels"]
    ]
    predictions_df = pd.DataFrame(
        prediction_list, columns=["file_id", "start", "end", "text", "label"]
    )

    if filter_labels:
        predictions_df = predictions_df[predictions_df["label"].isin(desired_columns)]

    # filter assertion labels
    assertion_labels = [
        "present",
        "absent",
        "possible",
        "conditional",
        "hypothetical",
        "associated_with_someone_else",
        "historical",
        "family",
    ]

    labels_df = labels_df[~labels_df["label"].isin(assertion_labels)]
    if filter_labels:
        labels_df = labels_df[labels_df["label"].isin(desired_columns)]
    predictions_df = predictions_df[~predictions_df["label"].isin(assertion_labels)]

    # rename label column to label_gold
    labels_df = labels_df.rename(columns={"label": "label_gold", "text": "text_gold"})
    # rename label column to label_pred
    predictions_df = predictions_df.rename(
        columns={"label": "label_pred", "text": "text_pred"}
    )

    # merged_df = pd.merge(labels_df, predictions_df, how="outer", on=["file_id", "start", "end"])
    # remove nan from the set
    labels_set = set(
        labels_df["label_gold"].tolist() + predictions_df["label_pred"].tolist()
    )
    labels_set = {label for label in labels_set if label == label}
    labels_set = sorted(labels_set)

    tp, fp, fn, classification_report = classify_matches(labels_df, predictions_df)

    # merged_df["correct"] = merged_df["label_gold"] == merged_df["label_pred"]

    classification_report_df = pd.DataFrame(classification_report)
    classification_report_df.to_csv(
        f"artifacts/results/cervical_classification_report_{eval_set}.csv",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )

    columns = [
        "label",
        "true_positives",
        "false_positives",
        "false_negatives",
        "support",
        "precision",
        "recall",
        "f1",
    ]

    metrics_list = []

    for label in labels_set:
        true_positives = len(
            classification_report_df[
                (classification_report_df["label"] == label)
                & (classification_report_df["classification"] == "true_positive")
            ]
        )
        false_positives = len(
            classification_report_df[
                (classification_report_df["label"] == label)
                & (classification_report_df["classification"] == "false_positive")
            ]
        )
        false_negatives = len(
            classification_report_df[
                (classification_report_df["label"] == label)
                & (classification_report_df["classification"] == "false_negative")
            ]
        )

        precision, recall, f1 = calculate_precision_recall_f1(
            true_positives, false_positives, false_negatives
        )

        label_metric = {
            "label": label,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "support": true_positives + false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        metrics_list.append(label_metric)

    true_positives = sum([m["true_positives"] for m in metrics_list])
    false_positives = sum([m["false_positives"] for m in metrics_list])
    false_negatives = sum([m["false_negatives"] for m in metrics_list])
    precision, recall, f1 = calculate_precision_recall_f1(
        true_positives, false_positives, false_negatives
    )

    micro_dict = {
        "label": "micro",
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "support": true_positives + false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # micro average
    macro_dict = {
        "label": "macro",
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "f1": np.mean([m["f1"] for m in metrics_list]),
    }

    results_per_label_df = pd.DataFrame(metrics_list, columns=columns)
    results_macro_micro_df = pd.DataFrame([micro_dict, macro_dict], columns=columns)

    results_per_label_df = results_per_label_df.round(3)
    results_macro_micro_df = results_macro_micro_df.round(3)

    results_per_label_df.to_csv(f"artifacts/results/cervical_eval_{eval_set}.csv")
    results_per_label_df.to_latex(f"artifacts/results/cervical_eval_{eval_set}.tex")

    results_macro_micro_df.to_csv(
        f"artifacts/results/cervical_micro_macro_{eval_set}.csv"
    )
    results_macro_micro_df.to_latex(
        f"artifacts/results/cervical_micro_macro_{eval_set}.tex"
    )

    results_df = pd.concat(
        [results_per_label_df, results_macro_micro_df], ignore_index=True
    )

    return results_df


def export_as_human_readable_format(doc_results_list, rule_set, out_dir):
    predicted_columns = [r.category for r in rule_set]

    desired_columns = [
        "Negative/Normal/NILM",
        "Atypical Squamous Cells of undetermined Significance",
        "Atypical squamous cells cannot exclude HSIL atypical squamous cells (ASC-H)",
        "Low-grade squamous intraepithelial lesion (LSIL)",
        "High-grade squamous intraepithelial lesion (HSIL)",
        "High-grade squamous intraepithelial lesion (HGSIL)",
        "Squamous Cell Carcinoma",
        "Atypical endocervical cells",
        "Atypical endometrial cells",
        "Atypical glandular cells",
        "Atypical glandular cells (favors neoplastic)",
        "Endocervical Adenocarcinoma in Situ (AIS)",
        "Adenocarcinoma",
        "Endocervical adenocarcinoma",
        "Endometrial adenocarcinoma",
        "Extrauterine adenocarcinoma",
        "Adenocarcinoma NOS",
        "HPV Negative",
        "HPV Positive",
        "HPV 16 Negative",
        "HPV 16 Positive",
        "HPV 18 Negative",
        "HPV 18 Positive",
        "HPV 18/45 Negative",
        "HPV 18/45 positive",
        "HPV Other Negative",
        "HPV Other positive",
    ]

    out_full_path = f"{out_dir}/cervical_notes_report_full.xlsx"
    out_subset_path = f"{out_dir}/cervical_notes_report_subset.xlsx"

    export_list = []

    for doc in doc_results_list:
        doc_row = doc["data"].copy()
        doc_results = doc["predictions"][0]["result"]
        for pred_column in predicted_columns:
            matching_results = next(
                filter(lambda x: pred_column in x["value"]["labels"], doc_results), None
            )
            if matching_results:
                doc_row[pred_column] = "Yes"
            else:
                doc_row[pred_column] = "No"
        export_list.append(doc_row)

    export_df = pd.DataFrame(export_list)

    # export_df.to_csv(out_file_with_pred_path, index=False)
    export_df.to_excel(out_full_path, index=False)

    subset_export_df = export_df[desired_columns]

    subset_export_df.to_excel(out_subset_path, index=False)


def record_performance_metrics(results_df, set_name, performance_baseline_dir):
    results_df.to_parquet(f"{performance_baseline_dir}/{set_name}_results.parquet")
    results_df.to_csv(f"{performance_baseline_dir}/{set_name}_results.csv", index=False)


def monitor_performance(
    results_df, set_name, performance_baseline_dir, rtol=0.01, atol=0.01
):
    # compare the result with the baseline, if the difference is greater than the tolerance, but less than twice
    # the tolerance, then we log a warning, if it is greater than twice the tolerance, we log an error
    baseline_results_df = pd.read_parquet(
        f"{performance_baseline_dir}/{set_name}_results.parquet"
    )

    logger.info(f"\n----\nMonitoring {set_name} performance")

    for _, row in results_df.iterrows():
        matching_baseline = baseline_results_df[
            (baseline_results_df["label"] == row["label"])
        ]

        if len(matching_baseline) != 1:
            logger.error(
                f"Unexpected number of matches for {row['label']}, len:{len(matching_baseline)}"
            )
            continue
        matching_baseline_row = matching_baseline.iloc[0]

        if row["f1"] != matching_baseline_row["f1"]:
            pass

        # we compare the f1 score
        if not np.allclose(
            row["f1"], matching_baseline_row["f1"], rtol=rtol, atol=atol
        ):
            if np.allclose(
                row["f1"], matching_baseline_row["f1"], rtol=2 * rtol, atol=2 * atol
            ):
                logger.error(
                    f"Error: {row['label']} f1 score changed from {matching_baseline_row['f1']} to {row['f1']}"
                )
            else:
                logger.warning(
                    f"Warning: {row['label']} f1 score changed from {matching_baseline_row['f1']} to {row['f1']}"
                )

    logger.info(f"Monitoring {set_name} performance done\n\n")


def batch_one_pipeline():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--record-performance",
        action="store_true",
        help="Record performance",
        default=False,
    )
    arg_parser.add_argument(
        "--performance-baseline-dir",
        type=str,
        help="Performance baseline directory",
        default="monitoring/20240829",
    )

    args = arg_parser.parse_args()

    record_performance = args.record_performance
    performance_baseline_dir = args.performance_baseline_dir

    start_timer = time.perf_counter()
    dataset_dict, notes_column = load_cervical_data()
    rule_set = load_cervical_rulebook()

    logger.info("Read file in {:.2f} seconds".format(time.perf_counter() - start_timer))
    start_timer = time.perf_counter()

    # train_results = run_medspacy_prediction_pipeline(
    #     dataset_dict["train"]["notes"], rule_set, notes_column
    # )

    valid_results = run_prediction_pipeline(
        dataset_dict["valid"]["notes"], rule_set, notes_column
    )

    logger.info(
        "Ran medspacy on valid in {:.2f} seconds".format(
            time.perf_counter() - start_timer
        )
    )

    test_1_results = run_prediction_pipeline(
        dataset_dict["test-1"]["notes"], rule_set, notes_column
    )

    logger.info(
        "Ran medspacy on test in {:.2f} seconds".format(
            time.perf_counter() - start_timer
        )
    )

    test_2_results = run_prediction_pipeline(
        dataset_dict["test-2"]["notes"], rule_set, notes_column
    )

    valid_results_df = evaluate_and_report(
        valid_results, dataset_dict["valid"]["labels"], eval_set="valid"
    )
    test_1_results_df = evaluate_and_report(
        test_1_results, dataset_dict["test-1"]["labels"], eval_set="test-1"
    )
    test_2_results_df = evaluate_and_report(
        test_2_results,
        dataset_dict["test-2"]["labels"],
        eval_set="test-2",
        filter_labels=True,
    )

    test_2_results_df = evaluate_and_report(
        test_2_results,
        dataset_dict["test-2"]["labels"],
        eval_set="test-2-full",
        filter_labels=False,
    )

    if record_performance:
        record_performance_metrics(valid_results_df, "valid", performance_baseline_dir)
        record_performance_metrics(
            test_1_results_df, "test-1", performance_baseline_dir
        )

    monitor_performance(valid_results_df, "valid", performance_baseline_dir)
    monitor_performance(test_1_results_df, "test-1", performance_baseline_dir)

    # export_as_label_studio_format(
    #     selected_notes, "artifacts/results/", remove_predictions=True
    # )
    # export_as_human_readable_format(test_1_results, rule_set, "artifacts/results/")

    # export_as_label_studio_format(
    #     test_2_results, "artifacts/results/", "test_2_results.json"
    # )

    logger.info("Evaluated in {:.2f} seconds".format(time.perf_counter() - start_timer))


def run_llm_analysis(results_list, notes_column):
    # Step 1: Extract annotations from results
    def extract_annotations(results):
        annotations = []
        for result in results:
            # Skip if no valid predictions
            if not result.get("predictions") or not result["predictions"][0].get(
                "result"
            ):
                continue

            for annotation in result["predictions"][0]["result"]:
                # Skip invalid annotations
                if not annotation.get("value") or "labels" not in annotation["value"]:
                    continue

                annotations.append(
                    {
                        "note_id": result["id"],
                        "prediction_id": annotation["id"],
                        "excerpt": result["data"].get(notes_column, ""),
                        "text": annotation["value"]["text"],
                        "label": annotation["value"]["labels"][0],
                    }
                )
        return annotations

    # Step 2: Run LLM analysis for assertions
    def run_llm_pipeline(annotations):
        pipeline_config = PipelineConfig(
            model="azure/gpt-4.1-mini",
            temperature=0,
            max_tokens=None,
            timeout=30,
            prompt_path="src/rulebook/prompts/assertion.txt",
            num_retries=3,
        )

        pipeline = StructuredLLMPipeline(
            config=pipeline_config,
            output_schema=ConceptExtractionVerification,
        )

        results = pipeline.process_items(annotations)

        # Add prediction_id to results
        return [
            {**r.model_dump(), "prediction_id": annotations[i]["prediction_id"]}
            if r is not None
            else None
            for i, r in enumerate(results)
        ]

    # Step 3: Merge modifier results back into original structure
    def merge_results(original_results, llm_results):
        # Create lookup dictionary for quick access
        assertion_lookup = {r["prediction_id"]: r for r in llm_results if r is not None}

        # Merge back into original structure
        for result in original_results:
            if not "predictions" in result:
                result["predictions"] = []

            prediction = {"model_version": "combined_model_v1", "result": []}

            # Process each annotation
            for annotation in result.get("predictions", [{}])[0].get("result", []):
                if not annotation.get("value"):
                    continue

                prediction_id = annotation["id"]

                # Keep the original label prediction
                prediction["result"].append(annotation)

                # Add corresponding assertion if available
                if prediction_id in assertion_lookup:
                    assertion_result = {
                        "value": {
                            "start": annotation["value"]["start"],
                            "end": annotation["value"]["end"],
                            "text": annotation["value"]["text"],
                            "labels": [assertion_lookup[prediction_id]["assertion"]],
                        },
                        "id": f"{prediction_id}",
                        "from_name": "assertion",
                        "to_name": "text",
                        "type": "labels",
                        "origin": "prediction",
                    }
                    prediction["result"].append(assertion_result)

            result["predictions"] = [prediction]

        return original_results

    # Main execution
    annotations = extract_annotations(results_list)
    llm_results = run_llm_pipeline(annotations)
    final_results = merge_results(results_list, llm_results)

    # Save results
    with open("artifacts/results/llm_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    return final_results


def batch_two_pipeline():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--record-performance",
        action="store_true",
        help="Record performance",
        default=False,
    )
    arg_parser.add_argument(
        "--performance-baseline-dir",
        type=str,
        help="Performance baseline directory",
        default="monitoring/20240829",
    )

    args = arg_parser.parse_args()

    record_performance = args.record_performance
    performance_baseline_dir = args.performance_baseline_dir

    start_timer = time.perf_counter()
    dataset_dict, notes_column = load_cervical_data_latest()
    rule_set = load_cervical_rulebook()

    logger.info("Read file in {:.2f} seconds".format(time.perf_counter() - start_timer))
    start_timer = time.perf_counter()

    # train_results = run_medspacy_prediction_pipeline(
    #     dataset_dict["train"]["notes"], rule_set, notes_column
    # )
    # select 10 notes randomly from valid
    selected_ids = [138, 314, 602, 2076, 4971, 6224, 8509, 12762]
    # dataset_dict["valid"]["notes"] = dataset_dict["valid"]["notes"].sample(10)
    debug_notes = dataset_dict["valid"]["notes"][
        dataset_dict["valid"]["notes"]["note_id"].isin(selected_ids)
    ]

    debug_results = run_prediction_pipeline(debug_notes, rule_set, notes_column)
    debug_llm_results = run_llm_analysis(debug_results, notes_column)

    export_as_label_studio_format(
        debug_llm_results,
        "artifacts/results/",
        "batch_four_debug.json",
        remove_predictions=False,
    )

    logger.info(
        "Ran medspacy on debug in {:.2f} seconds".format(
            time.perf_counter() - start_timer
        )
    )
    valid_results = run_prediction_pipeline(
        dataset_dict["valid"]["notes"], rule_set, notes_column
    )
    valid_llm_results = run_llm_analysis(valid_results, notes_column)

    export_as_label_studio_format(
        valid_llm_results,
        "artifacts/results/",
        "batch_four_validation.json",
        remove_predictions=False,
    )

    return

def main():
    # batch_one_pipeline()
    batch_two_pipeline()


if __name__ == "__main__":
    main()
    logger.info("Done")
