import functools
import hashlib
import json
import time

import medspacy
import numpy as np
import pandas as pd
from medspacy.ner import TargetRule

import parallel_helper
from resources import cervical_rulebook


def read_cervical_data():
    data_path = "datasets/cervical/data_with_phi/holt_2023_00185_pap_hpv_deid.csv"
    data_df = pd.read_csv(data_path)

    duplicate_notes_condition = data_df.duplicated(keep=False)
    data_df[duplicate_notes_condition].to_csv("artifacts/results/cervical_duplicate_notes.csv", index=False)

    data_df["note_id"] = range(len(data_df))
    data_df["note_hash"] = [hashlib.md5(note.encode()).hexdigest() for note in data_df["PROC_NARRATIVE"]]

    data_df = data_df[~duplicate_notes_condition]

    data_df.fillna(value="Not Available", inplace=True)

    notes_column = "PROC_NARRATIVE_CLEAN"  # "Proc narrative"

    data_df["PROC_NARRATIVE_CLEAN"] = data_df["PROC_NARRATIVE"].apply(lambda x: x.replace("\\n", "\n"))

    rule_set = [TargetRule(literal=rule.get("literal", ""), category=rule.get("category", ""),
                           pattern=rule.get("pattern", None))
                for rule in cervical_rulebook.cervical_rulebook_definition]

    return data_df, rule_set, notes_column


def get_medspacy_label(ent):
    context_to_i2b2_label_map = {
        "NEGATED_EXISTENCE": "absent",
        'POSSIBLE_EXISTENCE': "possible",
        "CONDITIONAL_EXISTENCE": "conditional",
        "HYPOTHETICAL": "hypothetical",
        'HISTORICAL': "historical",
        'FAMILY': "associated_with_someone_else"
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
            label = "associated_with_someone_else"

    return label


def run_medspacy_on_note(i_data_row, medspacy_nlp, note_text_column):
    i, data_row = i_data_row
    doc = medspacy_nlp(data_row[note_text_column])

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
        "data": {
            "num_predictions": len(prediction_list),
            **row_dict
        },
        "predictions": [
            {
                "result": prediction_list,
            }
        ]
    }

    return results_dict


def run_medspacy(notes_df, rule_set, notes_column):
    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(rule_set)

    run_medspacy_on_note_partial = functools.partial(run_medspacy_on_note,
                                                     medspacy_nlp=nlp,
                                                     note_text_column=notes_column)
    doc_results_list = parallel_helper.run_in_parallel_cpu_bound(run_medspacy_on_note_partial,
                                                                 notes_df.iterrows(),
                                                                 total=len(notes_df),
                                                                 max_workers=16)

    return doc_results_list


def export_results(doc_results_list, labels_list, out_dir):
    result_list = [result for doc in doc_results_list for result in doc if len(doc) > 0]

    results_df = pd.DataFrame(result_list)

    results_df.to_csv("artifacts/results/cervical_notes.csv", index=False)

    match_pattern_counts = results_df[["matched_pattern"]].groupby("matched_pattern").size().reset_index(name="count")

    missing_patterns = []
    for label, pattern in labels_list:
        if pattern not in match_pattern_counts["matched_pattern"].values:
            missing_patterns.append((pattern, 0))

    missing_patterns_df = pd.DataFrame(missing_patterns, columns=["matched_pattern", "count"])
    match_pattern_counts = pd.concat([match_pattern_counts, missing_patterns_df])

    match_pattern_counts = match_pattern_counts.sort_values(by="count", ascending=False)

    match_pattern_counts.to_csv("artifacts/results/cervical_notes_match_pattern_counts.csv", index=False)

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


def evaluate(predictions, labels_list):
    predictions = sorted(predictions, key=lambda x: (x[0], x[1], x[2]))
    labels_list = sorted(labels_list, key=lambda x: (x[0], x[1], x[2]))

    predictions_df = pd.DataFrame(predictions, columns=["file_name", "start", "end", "label"])
    labels_df = pd.DataFrame(labels_list, columns=["file_name", "start", "end", "label"])

    # rename label column to label_gold
    labels_df = labels_df.rename(columns={"label": "label_gold"})
    # rename label column to label_pred
    predictions_df = predictions_df.rename(columns={"label": "label_pred"})

    merged_df = pd.merge(labels_df, predictions_df, how="left", on=["file_name", "start", "end"])

    for i, row in merged_df.iterrows():
        matching_rows = merged_df.loc[(merged_df["file_name"] == row["file_name"]) &
                                      (merged_df["start"] == row["start"]) &

                                      (merged_df["end"] == row["end"])]
        if len(matching_rows) > 1:
            print(f"Found {len(matching_rows)} matching rows")
            print(matching_rows)
            print(row)
            print("\n\n")
            merged_df = merged_df.drop(matching_rows.index[1:])

    labels_metrics = {}
    label_set = list(set([l[3] for l in labels_list]))
    label_set = sorted(label_set)

    for label in label_set:
        true_positives = np.count_nonzero(
            np.logical_and(merged_df["label_pred"] == label, merged_df["label_gold"] == label))
        false_positives = np.count_nonzero(
            np.logical_and(merged_df["label_pred"] == label, merged_df["label_gold"] != label))
        false_negatives = np.count_nonzero(
            np.logical_and(merged_df["label_pred"] != label, merged_df["label_gold"] == label))

        precision, recall, f1 = calculate_precision_recall_f1(true_positives, false_positives, false_negatives)

        labels_metrics[label] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    true_positives = sum([v["true_positives"] for k, v in labels_metrics.items()])
    false_positives = sum([v["false_positives"] for k, v in labels_metrics.items()])
    false_negatives = sum([v["false_negatives"] for k, v in labels_metrics.items()])
    precision, recall, f1 = calculate_precision_recall_f1(true_positives, false_positives, false_negatives)

    labels_metrics["micro"] = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # micro average
    labels_metrics["macro"] = {
        "precision": np.mean([v["precision"] for k, v in labels_metrics.items()]),
        "recall": np.mean([v["recall"] for k, v in labels_metrics.items()]),
        "f1": np.mean([v["f1"] for k, v in labels_metrics.items()])
    }

    result_df = pd.DataFrame(labels_metrics).reset_index(names="label")

    column_order = ["present", "absent", "possible", "conditional", "hypothetical", "associated_with_someone_else",
                    "micro", "macro"]
    result_df = result_df[column_order]

    result_df = result_df.round(3)

    result_df.to_csv("artifacts/results/medspacy_context.csv")
    result_df.to_latex("artifacts/results/medspacy_context.tex")

    return result_df


def export_as_label_studio_format(doc_results_list, out_dir):
    out_file_path = f"{out_dir}/cervical_notes_with_labels.json"
    results_list = [doc for doc in doc_results_list if len(doc) > 0]
    with open(out_file_path, "w") as f:
        json.dump(results_list, f, indent=2)


def misc_func():
    export_as_label_studio_format(doc_results_list, "artifacts/results/")

    from collections import Counter

    l = Counter([r["value"]["text"] for p in doc_results_list for r in p["predictions"][0]["result"]])
    l = sorted(l.items(), key=lambda x: x[1], reverse=True)
    print(l)
    labels = set([ll for p in doc_results_list for r in p["predictions"][0]["result"] for ll in r["value"]["labels"]])

    known_labels = [r.category for r in rule_set]

    missing_labels = [l for l in known_labels if l not in labels]
    print(missing_labels)


def evaluate():
    results_df.rename(columns={"label": "pred_label"}, inplace=True)

    results_df_joined = results_df.merge(cervical_labels, on=["prediction_id", "matched_pattern", "pred_label"],
                                         how="left")

    eval_df = results_df_joined[["pred_label", "gold_label"]].groupby(
        ["pred_label", "gold_label"]).size().reset_index(name="count")

    micro_average = eval_df.groupby("gold_label").agg(
        count=("count", "sum")).reset_index()
    annotation_values = ["false positive", "false negative", "true positive"]

    for annotation in annotation_values:
        if annotation not in micro_average["gold_label"].values:
            micro_average.loc[len(micro_average)] = [annotation, 0]

    micro_average["pred_label"] = "micro_average"

    # concatenate the two dataframes
    eval_df = pd.concat([eval_df, micro_average], ignore_index=True)
    confusion_matrix_df = eval_df.pivot(index="pred_label", columns="gold_label", values="count").fillna(0)

    confusion_matrix_df.to_csv("artifacts/results/cervical_confusion_matrix.csv")
    results_df_joined.to_csv("artifacts/results/cervical_notes_with_labels.csv", index=False)
    results_df_joined.to_excel("artifacts/results/cervical_notes_with_labels.xlsx", index=False)
    results_labeled = results_df_joined.dropna(subset=["gold_label"])

    labels_count = results_labeled[["matched_pattern", "pred_label"]].groupby(
        ["matched_pattern", "pred_label"]).size().reset_index(name="count")

    labels_count.to_csv("artifacts/results/cervical_labels_count.csv", index=False)


def main():
    run_time = time.strftime("%Y%m%d-%H%M%S")
    notes_df, rule_set, notes_column = read_cervical_data()

    cervical_labels = pd.read_csv("datasets/cervical/cervical_labels.csv")
    labeled_notes = set(cervical_labels["prediction_id"].apply(lambda x: int(x.split("_")[1])))

    notes_df = notes_df[notes_df["note_id"].isin(labeled_notes)]

    # slice the notes
    # notes_df = notes_df.head(100)
    doc_results_list = run_medspacy(notes_df, rule_set, notes_column)

    export_as_label_studio_format(doc_results_list, "artifacts/results/")


if __name__ == '__main__':
    main()
