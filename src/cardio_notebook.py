"""
# ! pip install -q numpy medspacy tqdm spacy
# dbutils.library.restartPython()
"""

import concurrent.futures
import csv
import datetime
import functools
import hashlib
import json
import os
import re

import medspacy
import numpy as np
import pandas as pd
from medspacy.ner import TargetRule
from tqdm import tqdm

CARDIO_RULEBOOK = [
    {
        "category": "Malignant Neoplasm of the Breast",
        "pattern": r"breast cancer|chemo|chemotherapy|intraductal carcinoma|(malignant )?neoplasm"
    },
    {
        "category": "Melanoma",
        "pattern": r"skin cancer"
    },
    {
        "category": "Lung Cancer",
        "pattern": r"lung adenocarcinoma|squamous cell lung carcinoma|lung cancer|Neoplasm|chemotherapy"
    },
    {
        "category": "Kidney Cancer",
        "pattern": r"renal cancer"
    },
    {
        "category": "B-Cell Lymphoma",
        "pattern": r"hematological malignancy|chemotherapy|chemo|Neoplasm|chemotherapy|R-CHOP"
    },
    {
        "category": "Radiation (breast)",
        "pattern": r"radiation (breast)"
    },
    {
        "category": "Systolic Heart Failure",
        "pattern": r"Systolic heart failure|CHF|chronic systolic \(congestive\) heart failure|Congestive heart failure"
    },
    {
        "category": "Ischemic Cardiomyopathy and other cardiomyopathy",
        "pattern": r"Ischemic Cardiomyopathy and other cardiomyopathy"
    },
    {
        "category": "Coronary Artery Disease",
        "pattern": r"atherosclerosis"
    },
    {
        "category": "Diastolic Heart Failure",
        "pattern": r"diagstolic heart failure|diastolic dysfunction|diastolic heart failure|hypokinesis|CHF|\(?congestive\)? heart failure"
        # hypokinesis
        # b/l LE pitting edema
        # (congestive) heart failure
        # Congestive heart failure
        # CHF
        # (congestive) heart failure
        # Congestive heart failure
        # CHF
    },
    {
        "category": "Severe aortic stenosis",
        "pattern": r"severe aortic stenosis"
    },
    {
        "category": "Severe mitral regurgitation",
        "pattern": r"severe mitral regurgitation"
    },
    {
        "category": "Atrial fibrillation (ICD-10 I-48 group)",
        "pattern": r"arrhythmia"
    },
    {
        "category": "Severe pulmonary hypertension (RVSP > 60)",
        "pattern": r"severe pulmonary hypertension"
    },
    {
        "category": "Past Myocarditis",
        "pattern": r"past myocarditis"
    },
    {
        "category": "A1C > 9",
        "pattern": r"diabetes mellitus"
    },
    {
        "category": "Troponin > 0.02",
        "pattern": r"AMI|NSTEMI|Acute Myocardial Infarction|STEMI"
    },
    {
        "category": "Global Longitudinal Strain",
        "pattern": r"subclinical LV dysfunction|global longitudinal strain|GLS"
    },
    {
        "category": "Essential Hypertension",
        "pattern": r"essential hypertension"
    },
    {
        "category": "Diabetes Mellitus (1+2, exclude gestational and hospice)",
        "pattern": r"hyperglycemia"
    },
    {
        "category": "Age > 65",
        "pattern": r"geriatric|senior|advanced age"
    },
    {
        "category": "LDL =>190 and/or Xanthoma",
        "pattern": r"hyperlipidemia, mixed|familiar hyperlipidemia"
    },
    {
        "category": "HDL =<40",
        "pattern": r"dyslipidemia"
    },
    {
        "category": "BMI > 35",
        "pattern": r"obesity"
    },
    {
        "category": "EF 51%-54%",
        "pattern": r"(left ventricular |reduced )?ejection fraction|(LV|(?<= ))EF(?= )"
    },
    {
        "category": "Blood Pressure >= 140/90",
        "pattern": r"hypertension"
    },
    {
        "category": "Pro-BNP >= 400",
        "pattern": r"congestive heart failure|fluid retention"
    },
    {
        "category": "BNP > 100",
        "pattern": r"congestive heart failure|fluid retention"
    },
    {
        "category": "Current Smoker",
        "pattern": r"current smoker"
    },
    {
        "category": "Hyperlipidemia",
        "pattern": r"dyslipidemia|familiar hyperlipidemia|hyperlipidemia|hypertriglyceridemia"
    },
    {
        "category": "African-American race",
        "pattern": r"african-american"
    },
    {
        "category": "LDL 160 to 189",
        "pattern": r"hyperlipidemia, mixed"
    },
    {
        "category": "HDL 41 to 59",
        "pattern": r"dyslipidemia"
    },
    {
        "category": "Former Smoker",
        "pattern": r"former smoker"
    },
    {
        "category": "CART Cell Procedure",
        "pattern": r"bone marrow transplantation"
    }
]


def get_ruleset():
    rule_set = [TargetRule(literal=rule.get("literal", ""),
                           category=rule.get("category", ""),
                           pattern=rule.get("pattern", None))
                for rule in CARDIO_RULEBOOK]

    return rule_set


def run_in_parallel_cpu_bound(func, iterable, max_workers=None, disable=False, total=None, **kwargs):
    """
    Run a function in parallel on a list of arguments
    :param disable: whether to disable the progress bar
    :param func: function to run
    :param iterable: list of arguments
    :param max_workers: maximum number of workers to use
    :param kwargs: keyword arguments to pass to the function
    :return: list of results
    """
    results = []
    if hasattr(iterable, "len"):
        total = len(iterable)

    with tqdm(total=total, disable=disable) as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_dict = {
                executor.submit(func, item, **kwargs): index
                for index, item in enumerate(iterable)
            }
            for future in concurrent.futures.as_completed(future_dict):
                results.append(future.result())
                progress_bar.update(1)
    return results


def get_medspacy_label(ent):
    context_to_i2b2_label_map = {
        "NEGATED_EXISTENCE": "absent",
        'POSSIBLE_EXISTENCE': "possible",
        "CONDITIONAL_EXISTENCE": "conditional",
        "HYPOTHETICAL": "hypothetical",
        'HISTORICAL': "historical",
        'FAMILY': "family"
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



def parse_value_from_sentence(sentence_text, label, matched_pattern):
    value = None
    if label in ["Global Longitudinal Strain",
                 "EF 51%-54%", ]:

        pattern_index_in_sentence = sentence_text.find(matched_pattern)
        sentence_starting_with_matched_pattern = sentence_text[pattern_index_in_sentence:]
        for pattern in value_patterns:
            match = pattern.search(sentence_starting_with_matched_pattern)
            if match:
                value = match.group(1)
                break
    return value


def run_medspacy_on_note(i_data_row, medspacy_nlp, note_text_column, out_dir):
    i, data_row = i_data_row
    row_dict = data_row.to_dict()
    note_id = data_row["note_id"]
    #
    if note_id == 2:
        pass

    doc = medspacy_nlp(data_row[note_text_column])

    value_pattern = re.compile(r'(\d{1,2}-\d{1,2})%|(\d{1,2}(\.\d{1,2})?%)')

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

        if label in ["Global Longitudinal Strain",
                     "EF 51%-54%", ]:
            match = value_pattern.search(doc.text[ent.start_char:len(doc.text)])
            if match:
                value = match[0]
                value_start = ent.start_char + match.start()
                value_end = ent.start_char + match.end()
                value_label = "GLS Value" if label == "Global Longitudinal Strain" else "EF Value"
                prediction_id = f"note_{note_id}_ent_{value_start}_{value_end}"
                prediction_dict = {
                    "value": {
                        "start": value_start,
                        "end": value_end,
                        "text": value,
                        "labels": [value_label],
                    },
                    "id": prediction_id,
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "origin": "prediction",
                }
                prediction_list.append(prediction_dict)
            # else:
            #     print(f"Value not found for {label} in note {note_id}, text: {doc.text[ent.start_char:]}")

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


def get_notes_df(spark):
    ids_list = get_ids_of_interest(spark)

    in_clause = ",".join(map(lambda x: f"'{x}'", ids_list))
    table_name = "`hive_metastore`.`cardio_oncology`.`unstructured_notes`"
    total_rows = spark.sql(f"SELECT COUNT(*) FROM {table_name} WHERE PAT_ID IN ({in_clause})").collect()[0][0]

    query = f"SELECT * FROM {table_name} WHERE PAT_ID IN ({in_clause})"
    notes_spark = spark.sql(query)

    notes_df = notes_spark.toPandas()

    return notes_df


def get_ids_of_interest(spark):
    ids_table_name = "`hive_metastore`.`cardio_oncology`.`ids_of_interest`"

    query = f"SELECT * FROM {ids_table_name}"
    ids_spark = spark.sql(query)

    ids_df = ids_spark.toPandas()

    ids_list = ids_df["EpicPatientID"].unique().tolist()

    return ids_list


def sample_results(doc_results_list, k):
    desired_labels = ["Global Longitudinal Strain",
                      "EF 51%-54%", ]

    doc_results_list = [doc for doc in doc_results_list if [r for r in doc["predictions"] for rr in r["result"] if
                                                            rr["value"]["labels"][0] in desired_labels]]
    selected_results = []
    selected_results_label_count_dict = {}
    text_to_notes_dict = {}
    label_to_notes_dict = {}
    for i, doc in enumerate(doc_results_list):
        for prediction in doc["predictions"]:
            result = prediction["result"]

            for r in result:
                text = r["value"]["text"]
                label = r["value"]["labels"][0]
                if text not in text_to_notes_dict:
                    text_to_notes_dict[text] = []
                text_to_notes_dict[text].append(i)
                if label not in label_to_notes_dict:
                    label_to_notes_dict[label] = []
                label_to_notes_dict[label].append(i)

    text_to_notes_dict = dict(sorted(text_to_notes_dict.items(), key=lambda x: len(x[1])))
    label_to_notes_dict = dict(sorted(label_to_notes_dict.items(), key=lambda x: len(x[1])))
    selected_notes = []
    for label, notes in label_to_notes_dict.items():
        for note_index in notes:
            if note_index not in selected_notes:
                selected_notes.append(note_index)
            break
    for label, notes in label_to_notes_dict.items():
        for note_index in notes:
            if note_index not in selected_notes:
                selected_notes.append(note_index)
    sampled_results = [doc_results_list[i] for i in selected_notes[:k]]
    return sampled_results


def report_results_label_count(doc_results_list, file_name):
    labels_list = [l["category"] for l in CARDIO_RULEBOOK]
    label_to_count_dict = {l: 0 for l in labels_list}
    for doc in doc_results_list:
        for prediction in doc["predictions"]:
            result = prediction["result"]
            for r in result:
                label = r["value"]["labels"][0]
                if label not in label_to_count_dict:
                    label_to_count_dict[label] = 0
                label_to_count_dict[label] += 1
    label_count_df = pd.DataFrame(label_to_count_dict.items(), columns=["label", "count"])
    label_count_df = label_count_df.sort_values("count", ascending=True)

    label_count_df.to_csv(f"{file_name}", index=False)


def export_as_label_studio_format(doc_results_list, out_dir):
    out_file_path = f"{out_dir}/cardio_notes.json"
    results_list = [doc for doc in doc_results_list if len(doc) > 0]
    with open(out_file_path, "w") as f:
        json.dump(results_list, f, indent=2)


def run_medspacy(notes_df, rule_set, notes_column, out_dir):
    nlp = medspacy.load(medspacy_enable=[
        "medspacy_pyrush",
        "medspacy_target_matcher",
        "medspacy_context"
    ])
    # nlp.add_pipe(
    #     "medspacy_target_matcher",
    #     config={
    #         "phrase_matcher_attr": "LOWER",
    #         "prune": False,
    #     }
    #
    # )
    # matcher = MedspacyMatcher(nlp, prune=False)
    # matcher = MedspacyMatcher(nlp, name="medspacy_target_matcher", phrase_matcher_attr="LOWER", prune=False)

    # nlp.add_pipe(matcher)
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(rule_set)
    target_matcher._prune = False
    # nlp = matcher

    notes_df["note_hash"] = notes_df["NOTE_TEXT"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    # notes_df["note_id"] = range(len(notes_df))

    run_medspacy_on_note_partial = functools.partial(run_medspacy_on_note,
                                                     medspacy_nlp=nlp,
                                                     note_text_column=notes_column,
                                                     out_dir=out_dir)
    doc_results_list = run_in_parallel_cpu_bound(run_medspacy_on_note_partial,
                                                 notes_df.iterrows(),
                                                 total=len(notes_df),
                                                 # max_workers=16
                                                 )

    return doc_results_list


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


def evaluate_v1(doc_results_list, cervical_labels, out_name="cardio_eval"):
    prediction_list = [(doc["data"]["note_id"], r["value"]["start"], r["value"]["end"], r["value"]["text"], rr) for doc
                       in
                       doc_results_list for
                       prediction in doc["predictions"] for r in prediction["result"] for rr in r["value"]["labels"]]
    gold_labels = [(l["data"]["note_id"], rr["value"]["start"], rr["value"]["end"], rr["value"]["text"], rrr) for l in
                   cervical_labels for r
                   in l["annotations"] for rr in r["result"] for rrr in rr["value"]["labels"]]

    # filter assertion labels
    assertion_labels = ["present", "absent",
                        "possible", "conditional",
                        "hypothetical", "associated_with_someone_else",
                        "historical", "family",
                        "negated_existence", "possible_existence",
                        ]

    assertion_upper = [a.upper() for a in assertion_labels]

    assertion_labels += assertion_upper

    # prediction_list = sorted(prediction_list)
    # gold_labels = sorted(gold_labels)

    predictions_df = pd.DataFrame(prediction_list, columns=["file_id", "start", "end", "text", "label"])
    labels_df = pd.DataFrame(gold_labels, columns=["file_id", "start", "end", "text", "label"])

    labels_df = labels_df[~labels_df["label"].isin(assertion_labels)]
    predictions_df = predictions_df[~predictions_df["label"].isin(assertion_labels)]

    # rename label column to label_gold
    labels_df = labels_df.rename(columns={"label": "label_gold", "text": "text_gold"})
    # rename label column to label_pred
    predictions_df = predictions_df.rename(columns={"label": "label_pred", "text": "text_pred"})

    merged_df = pd.merge(labels_df, predictions_df, how="outer", on=["file_id", "start", "end"])

    merged_df["correct"] = merged_df["label_gold"] == merged_df["label_pred"]

    merged_df.to_csv(f"artifacts/results/{out_name}_all.csv", index=False,
                     quoting=csv.QUOTE_NONNUMERIC)

    # for i, row in merged_df.iterrows():
    #     matching_rows = merged_df.loc[(merged_df["file_id"] == row["file_id"]) &
    #                                   (merged_df["start"] == row["start"]) &
    #                                   (merged_df["end"] == row["end"])]
    #     if len(matching_rows) > 1:
    #         print(f"Found {len(matching_rows)} matching rows")
    #         print(matching_rows)
    #         print(row)
    #         print("\n\n")
    #         merged_df = merged_df.drop(matching_rows.index[1:])

    labels_set = set(merged_df["label_gold"].tolist() + merged_df["label_pred"].tolist())

    # remove nan from the set
    labels_set = {label for label in labels_set if label == label}
    labels_set = sorted(labels_set)

    columns = ["label", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1"]

    metrics_list = []

    for label in labels_set:
        true_positives = np.count_nonzero(
            np.logical_and(merged_df["label_pred"] == label, merged_df["label_gold"] == label))
        false_positives = np.count_nonzero(
            np.logical_and(merged_df["label_pred"] == label, merged_df["label_gold"] != label))
        false_negatives = np.count_nonzero(
            np.logical_and(merged_df["label_pred"] != label, merged_df["label_gold"] == label))

        precision, recall, f1 = calculate_precision_recall_f1(true_positives, false_positives, false_negatives)

        label_metric = {
            "label": label,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        metrics_list.append(label_metric)

    true_positives = sum([m["true_positives"] for m in metrics_list])
    false_positives = sum([m["false_positives"] for m in metrics_list])
    false_negatives = sum([m["false_negatives"] for m in metrics_list])
    precision, recall, f1 = calculate_precision_recall_f1(true_positives, false_positives, false_negatives)

    micro_dict = {
        "label": "micro",
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # micro average
    macro_dict = {
        "label": "macro",
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "f1": np.mean([m["f1"] for m in metrics_list])
    }

    metrics_list.append(micro_dict)
    metrics_list.append(macro_dict)

    result_df = pd.DataFrame(metrics_list, columns=columns)

    result_df = result_df.round(3)

    result_df.to_csv(f"artifacts/results/{out_name}.csv")
    result_df.to_latex(f"artifacts/results/{out_name}.tex")

    return result_df


# Function to find matches according to the new criteria
def classify_matches(labels_df, predictions_df):
    labels_df = labels_df.copy()
    predictions_df = predictions_df.copy()
    tp, fp, fn = 0, 0, 0
    classification_report = []
    # Check each prediction
    for _, pred_row in predictions_df.iterrows():
        match_found = False
        for index, true_row in labels_df.iterrows():
            if (pred_row['file_id'] == true_row['file_id'] and
                    pred_row['label_pred'] == true_row['label_gold'] and
                    do_intersect(pred_row['start'], pred_row['end'], true_row['start'], true_row['end'])):
                match_found = True
                labels_df.drop(index, inplace=True)  # Remove to avoid double counting
                tp += 1
                classification_report.append(
                    {'file_id': pred_row['file_id'],
                     'label': pred_row['label_pred'],
                     "classification": "true_positive",
                     "pred_start": pred_row['start'],
                     "pred_end": pred_row['end'],
                     "gold_start": true_row['start'],
                     "gold_end": true_row['end'],
                     "pred_text": pred_row['text_pred'],
                     "gold_text": true_row['text_gold']})
                break
        if not match_found:
            fp += 1
            classification_report.append({
                'file_id': pred_row['file_id'],
                'label': pred_row['label_pred'],
                "classification": "false_positive",
                "pred_start": pred_row['start'],
                "pred_end": pred_row['end'],
                "pred_text": pred_row['text_pred']
            })
    # Remaining entries are false negatives
    fn = len(labels_df)
    for _, fn_row in labels_df.iterrows():
        classification_report.append({
            'file_id': fn_row['file_id'],
            'label': fn_row['label_gold'],
            "classification": "false_negative",
            "gold_start": fn_row['start'],
            "gold_end": fn_row['end'],
            "gold_text": fn_row['text_gold']
        })

    return tp, fp, fn, classification_report

# Function to determine if two spans intersect
def do_intersect(start1, end1, start2, end2):
    return max(start1, start2) <= min(end1, end2)


def evaluate(doc_results_list, cervical_labels, eval_set="test"):
    prediction_list = [(doc["id"], r["value"]["start"], r["value"]["end"], r["value"]["text"], rr) for doc in
                       doc_results_list for
                       prediction in doc["predictions"] for r in prediction["result"] for rr in r["value"]["labels"]]
    gold_labels = [(l["data"]["note_id"], rr["value"]["start"], rr["value"]["end"], rr["value"]["text"], rrr) for l in
                   cervical_labels for r
                   in l["annotations"] for rr in r["result"] for rrr in rr["value"]["labels"]]

    # filter assertion labels
    assertion_labels = ["present", "absent",
                        "possible", "conditional",
                        "hypothetical", "associated_with_someone_else",
                        "historical", "family"]
    context_to_i2b2_label_map = {
        "NEGATED_EXISTENCE": "absent",
        'POSSIBLE_EXISTENCE': "possible",
        "CONDITIONAL_EXISTENCE": "conditional",
        "HYPOTHETICAL": "hypothetical",
        'HISTORICAL': "historical",
        'FAMILY': "family"
    }

    assertion_labels += list(context_to_i2b2_label_map.keys())

    # prediction_list = sorted(prediction_list)
    # gold_labels = sorted(gold_labels)

    predictions_df = pd.DataFrame(prediction_list, columns=["file_id", "start", "end", "text", "label"])
    labels_df = pd.DataFrame(gold_labels, columns=["file_id", "start", "end", "text", "label"])

    labels_df = labels_df[~labels_df["label"].isin(assertion_labels)]
    predictions_df = predictions_df[~predictions_df["label"].isin(assertion_labels)]

    # rename label column to label_gold
    labels_df = labels_df.rename(columns={"label": "label_gold", "text": "text_gold"})
    # rename label column to label_pred
    predictions_df = predictions_df.rename(columns={"label": "label_pred", "text": "text_pred"})

    # merged_df = pd.merge(labels_df, predictions_df, how="outer", on=["file_id", "start", "end"])
    # remove nan from the set
    labels_set = set(labels_df["label_gold"].tolist() + predictions_df["label_pred"].tolist())
    labels_set = {label for label in labels_set if label == label}
    labels_set = sorted(labels_set)

    tp, fp, fn, classification_report = classify_matches(labels_df, predictions_df)

    # merged_df["correct"] = merged_df["label_gold"] == merged_df["label_pred"]

    classification_report_df = pd.DataFrame(classification_report)
    classification_report_df.to_csv(f"artifacts/results/cervical_classification_report_{eval_set}.csv",
                                    index=False,
                                    quoting=csv.QUOTE_NONNUMERIC)
    # merged_df.to_csv(f"artifacts/results/cervical_eval_detail_{eval_set}.csv", index=False,
    #                  quoting=csv.QUOTE_NONNUMERIC)

    # for i, row in merged_df.iterrows():
    #     matching_rows = merged_df.loc[(merged_df["file_id"] == row["file_id"]) &
    #                                   (merged_df["start"] == row["start"]) &
    #                                   (merged_df["end"] == row["end"])]
    #     if len(matching_rows) > 1:
    #         print(f"Found {len(matching_rows)} matching rows")
    #         print(matching_rows)
    #         print(row)
    #         print("\n\n")
    #         merged_df = merged_df.drop(matching_rows.index[1:])

    columns = ["label", "true_positives", "false_positives", "false_negatives",
               "support", "precision", "recall", "f1"]

    metrics_list = []

    for label in labels_set:
        # true_positives = np.count_nonzero(
        #     np.logical_and(merged_df["label_pred"] == label, merged_df["label_gold"] == label))
        # false_positives = np.count_nonzero(
        #     np.logical_and(merged_df["label_pred"] == label, merged_df["label_gold"] != label))
        # false_negatives = np.count_nonzero(
        #     np.logical_and(merged_df["label_pred"] != label, merged_df["label_gold"] == label))

        true_positives = len(classification_report_df[(classification_report_df["label"] == label) &
                                                      (classification_report_df["classification"] == "true_positive")])
        false_positives = len(classification_report_df[(classification_report_df["label"] == label) &
                                                       (classification_report_df[
                                                            "classification"] == "false_positive")])
        false_negatives = len(classification_report_df[(classification_report_df["label"] == label) &
                                                       (classification_report_df[
                                                            "classification"] == "false_negative")])

        precision, recall, f1 = calculate_precision_recall_f1(true_positives, false_positives, false_negatives)

        label_metric = {
            "label": label,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "support": true_positives + false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        metrics_list.append(label_metric)

    true_positives = sum([m["true_positives"] for m in metrics_list])
    false_positives = sum([m["false_positives"] for m in metrics_list])
    false_negatives = sum([m["false_negatives"] for m in metrics_list])
    precision, recall, f1 = calculate_precision_recall_f1(true_positives, false_positives, false_negatives)

    micro_dict = {
        "label": "micro",
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "support": true_positives + false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # micro average
    macro_dict = {
        "label": "macro",
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "f1": np.mean([m["f1"] for m in metrics_list])
    }

    # metrics_list.append(micro_dict)
    # metrics_list.append(macro_dict)

    result_df = pd.DataFrame(metrics_list, columns=columns)
    micro_macro_df = pd.DataFrame([micro_dict, macro_dict], columns=columns)
    result_df = result_df.round(3)
    micro_macro_df = micro_macro_df.round(3)

    result_df.to_csv(f"artifacts/results/cardio_eval_{eval_set}.csv")
    result_df.to_latex(f"artifacts/results/cardio_eval_{eval_set}.tex")

    micro_macro_df.to_csv(f"artifacts/results/cardio_micro_macro_{eval_set}.csv")
    micro_macro_df.to_latex(f"artifacts/results/cardio_micro_macro_{eval_set}.tex")

    return result_df


def export_as_human_readable_format(doc_results_list, rule_set, out_dir):
    predicted_columns = [r.category for r in rule_set]

    out_file_path = f"{out_dir}/cardio_notes_human_readable.csv"
    out_file_with_pred_path = f"{out_dir}/cardio_notes_human_readable_with_pred.csv"

    export_list = []

    for doc in doc_results_list:
        doc_row = doc["data"].copy()
        doc_results = doc["predictions"][0]["result"]
        for pred_column in predicted_columns:
            matching_results = next(filter(lambda x: pred_column in x["value"]["labels"], doc_results), None)
            if matching_results:
                doc_row[pred_column] = "Yes"
            else:
                doc_row[pred_column] = "No"
        export_list.append(doc_row)

    export_df = pd.DataFrame(export_list)

    export_df.to_csv(out_file_with_pred_path, index=False)
    export_df.to_excel(out_file_with_pred_path.replace(".csv", ".xlsx"), index=False)


def cardio_oncology_pipeline(notes_df):
    project_dir = "/Workspace/Users/vamarvan23@osfhealthcare.org/"
    # remote dir shared
    current_datetime_str = datetime.datetime.now().strftime("%Y%m%d")
    out_dir = f"/Workspace/Shared/NLP/{current_datetime_str}"
    out_dir = f"{current_datetime_str}"

    valid_file_path = "datasets/cardio-oncology/cardio-validation.json"

    with open(valid_file_path, "r") as f:
        validation_labels = json.load(f)

    context_to_i2b2_label_map = {
        "NEGATED_EXISTENCE": "absent",
        'POSSIBLE_EXISTENCE': "possible",
        "CONDITIONAL_EXISTENCE": "conditional",
        "HYPOTHETICAL": "hypothetical",
        'HISTORICAL': "historical",
        'FAMILY': "family"
    }


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    rule_set = get_ruleset()

    valid_note_ids = [note["data"]["note_id"] for note in validation_labels]

    valid_set = notes_df[notes_df["note_id"].isin(valid_note_ids)]

    valid_results = run_medspacy(valid_set, rule_set, "NOTE_TEXT", out_dir)

    evaluate_v1(valid_results, validation_labels)

    valid_metrics = evaluate(valid_results, validation_labels, eval_set="valid")

    report_results_label_count(valid_results, out_dir + "/label_count.csv")

    sampled_results = sample_results(valid_results, 100)

    report_results_label_count(sampled_results, out_dir + "/sampled_label_count.csv")
    export_as_label_studio_format(sampled_results, out_dir)

    export_as_human_readable_format(valid_results, rule_set, "artifacts/results/")



def main():
    # notes_df = get_notes_df(spark)

    notes_df = pd.read_parquet("notes_df.parquet")
    # notes_df = notes_df.sample(1000)
    cardio_oncology_pipeline(notes_df)


if __name__ == "__main__":
    main()
