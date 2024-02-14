import csv
import functools
import time
import medspacy
import numpy as np
import pandas as pd
from medspacy.ner import TargetRule

import parallel_helper


def read_cervical_data():
    terms_file_path = "datasets/cervical/terms.csv"
    data_path_1 = "datasets/cervical/data_with_phi/holt_2023_00185_pap_hpv_deid.csv"
    # data_path_2 = "datasets/cervical/CCTSCRDWRequest20230_DATA_LABELS_2023-11-14_1224.csv"
    data_df = pd.read_csv(data_path_1)

    duplicate_notes_condition = data_df.duplicated(keep=False)
    data_df[duplicate_notes_condition].to_csv("artifacts/results/cervical_duplicate_notes.csv", index=False)

    data_df["note_id"] = range(len(data_df))

    data_df = data_df[~duplicate_notes_condition]

    label_pattern_list = []
    with open(terms_file_path, "r") as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if i == 0 or len(row) == 0:
                continue
            group, label, term_list = row[0], row[1], row[2:]
            if len(label.strip()) == 0:
                continue
            term_list = [term.strip().replace('"', '').replace("'", "") for term in term_list]
            term_list = [term for term in term_list if len(term) > 0]
            for term in term_list:
                label_pattern_list.append((label, term))
    notes_column = "PROC_NARRATIVE"  # "Proc narrative"

    return data_df, label_pattern_list, notes_column


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
    # row_dict.pop(note_text_column, None)
    results_list = []
    if len(doc.ents) > 0:
        for sentence in doc.sents:
            for ent in sentence.ents:
                sentence_text = sentence.text
                matched_pattern = ent.text
                label = ent.label_
                assertion = get_medspacy_label(ent)

                # ent_10_window_start = max(ent.start - 5, 0)
                # ent_10_window_end = min(ent.end + 5, len(sentence.doc))
                ent_20_window_start = max(ent.start - 10, 0)
                ent_20_window_end = min(ent.end + 10, len(sentence.doc))
                # window_context_10 = " ".join(
                #     [token.text for token in sentence.doc[ent_10_window_start:ent_10_window_end]])
                window_context_20 = " ".join(
                    [token.text for token in sentence.doc[ent_20_window_start:ent_20_window_end]])

                prediction_id = f"note_{note_id}_ent_{ent.start}_{ent.end}"

                results_dict = {
                    "context": window_context_20,
                    "matched_pattern": matched_pattern,
                    "label": label,
                    "assertion": assertion,
                    "prediction_id": prediction_id,
                    **row_dict
                }
                results_list.append(results_dict)

    return results_list


def run_medspacy(notes_df, labels_list, notes_column):
    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [TargetRule(literal=r[1].strip(), category=r[0]) for r in labels_list]
    target_matcher.add(target_rules)

    # notes_df = notes_df.head(100)

    run_medspacy_on_note_partial = functools.partial(run_medspacy_on_note,
                                                     medspacy_nlp=nlp,
                                                     note_text_column=notes_column)
    doc_list = parallel_helper.run_in_parallel_cpu_bound(run_medspacy_on_note_partial,
                                                         notes_df.iterrows(),
                                                         total=len(notes_df),
                                                         max_workers=16)

    result_list = [result for doc in doc_list for result in doc if len(doc) > 0]

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


# text = """
# Past Medical History:
# 1. Atrial fibrillation
# 2. Type II Diabetes Mellitus
#
# Assessment and Plan:
# There is no evidence of pneumonia. Continue warfarin for Afib. Follow up for management of type 2 DM.
# """


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

    print(result_df)


def main():
    run_time = time.strftime("%Y%m%d-%H%M%S")
    notes_df, labels_list, notes_column = read_cervical_data()
    #
    # # slice the first 1000 notes
    # # notes_df = notes_df.head(1000)
    #
    # run_medspacy(notes_df, labels_list, notes_column)
    #
    results_df = pd.read_csv("artifacts/results/cervical_notes.csv")

    results_df.rename(columns={"label": "pred_label"}, inplace=True)

    cervical_labels = pd.read_csv("datasets/cervical/cervical_labels.csv")

    results_df_joined = results_df.merge(cervical_labels, on=["prediction_id", "matched_pattern", "pred_label"],
                                         how="left")

    results_df_joined.to_csv("artifacts/results/cervical_notes_with_labels.csv", index=False)

    results_labeled = results_df_joined.dropna(subset=["gold_label"])

    labels_count = results_labeled[["matched_pattern", "pred_label"]].groupby(
        ["matched_pattern", "pred_label"]).size().reset_index(name="count")

    labels_count.to_csv("artifacts/results/cervical_labels_count.csv", index=False)

    # pivoting

    print("")


if __name__ == '__main__':
    main()

# def main():
#     labels_files = collections.Counter([l[0] for l in labels_list])
#     labels_files = sorted(labels_files.items(), key=lambda x: x[1], reverse=True)
#     selected_files = [l[0] for l in labels_files[:10]]
#     sliced_labels = [l for l in labels_list if l[0] in selected_files]
#     sliced_medspacy_doc_dict = {k: v for k, v in medspacy_doc_dict.items() if k in selected_files}
#
#     predictions = run_contex(sliced_medspacy_doc_dict)
#     evaluate(predictions, sliced_labels, "medspacy_context", sliced_medspacy_doc_dict)
#
#
# if __name__ == '__main__':
#     main()
