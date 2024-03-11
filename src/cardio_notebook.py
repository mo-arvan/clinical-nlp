"""
# ! pip install -q numpy medspacy tqdm spacy
# dbutils.library.restartPython()
"""

import concurrent.futures
import datetime
import functools
import hashlib
import json
import medspacy
import numpy as np
import os
import pandas as pd
import re
import time

from medspacy.ner import TargetRule
from tqdm import tqdm




CARDIO_RULEBOOK = [
    {
        "category": "Malignant Neoplasm of the Breast",
        "pattern": r"breast cancer"
    },
    {
        "category": "Melanoma",
        "pattern": r"skin cancer"
    },
    {
        "category": "Lung Cancer",
        "pattern": r"lung adenocarcinoma|squamous cell lung carcinoma"
    },
    {
        "category": "Kidney Cancer",
        "pattern": r"renal cancer"
    },
    {
        "category": "B-Cell Lymphoma",
        "pattern": r"hematological malignancy"
    },
    {
        "category": "Radiation (breast)",
        "pattern": r"radiation (breast)"
    },
    {
        "category": "Systolic Heart Failure",
        "pattern": r"reduced ejection fraction"
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
        "pattern": r"diagstolic heart failure"
    },
    {
        "category": "Severe aortic stenosis",
        "pattern": r"severe aortic stenosis"
    },
    {
        "category": "Severe mitral regurgitation",
        "pattern": r"svere mitral regurgitation"
    },
    {
        "category": "Atrial fibrillation (ICD-10 I-48 group)",
        "pattern": r"arrhythmia"
    },
    {
        "category": "Severe pulmonary hypertension (RVSP > 60)",
        "pattern": r"svere pulmonary hypertension"
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
        "category": "Diabetes Mellitus (1+2, exclude gestational & hospice)",
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
        "pattern": r"ejection fraction|LVEF|left ventricular ejection fraction"
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
        "pattern": r"dyslipidemia|familiar hyperlipidemia"
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

    modifiers_category = [mod.category for mod in ent._.modifiers]
    label = None
    if len(modifiers_category) == 0:
        # no modifiers, assume present
        label = "present"
    elif len(modifiers_category) == 1:
        # we have a single modifier, we report it
        label = modifiers_category[0]  # context_to_i2b2_label_map[modifiers_category[0]]
    else:
        modifiers_str = ", ".join(modifiers_category)
        # more than one modifier, we report the most frequent one
        # we decide an order of precedence
        # 1. absent
        # 2. possible
        # 3. hypothetical
        # 4. conditional
        # 5. not associated
        if ent._.is_negated:
            label = "POSSIBLE_EXISTENCE"
        elif ent._.is_uncertain:
            label = "POSSIBLE_EXISTENCE"
        # currently we cannot handle conditional
        # elif ent._.is_conditional:
        #     label = "conditional"
        elif ent._.is_hypothetical:
            label = "HYPOTHETICAL"
        # i2b2 does not have historical labels
        # elif ent._.is_historical:
        #     label = "historical"
        elif ent._.is_family:
            label = "FAMILY"

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


def run_medspacy_on_note(i_data_row, medspacy_nlp, note_text_column):
    i, data_row = i_data_row
    doc = medspacy_nlp(data_row[note_text_column])

    row_dict = data_row.to_dict()
    note_id = data_row["note_id"]

    value_pattern = re.compile(r'(\d{1,2}-\d{1,2})%|(\d{1,2})%')

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
            match = value_pattern.search(text[ent.start_char:])
            if match:
                value = match.group(1)
                value_start = ent.start_char + match.start()
                value_end = ent.start_char + match.end()
                prediction_id = f"note_{note_id}_ent_{value_start}_{value_end}"
                prediction_dict = {
                    "value": {
                        "start": ent.start_char + match.start(),
                        "end": ent.start_char + match.end(),
                        "text": value,
                        "labels": ["value"],
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

def export_as_label_studio_format(doc_results_list, out_dir):
    out_file_path = f"{out_dir}/cardio_notes.json"
    results_list = [doc for doc in doc_results_list if len(doc) > 0]
    with open(out_file_path, "w") as f:
        json.dump(results_list, f, indent=2)


def run_medspacy(notes_df, rule_set, notes_column):

    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(rule_set)

    notes_df["note_hash"] = notes_df["NOTE_TEXT"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    notes_df["note_id"] = range(len(notes_df))


    run_medspacy_on_note_partial = functools.partial(run_medspacy_on_note,
                                                     medspacy_nlp=nlp,
                                                     note_text_column=notes_column)
    doc_results_list = run_in_parallel_cpu_bound(run_medspacy_on_note_partial,
                                                                 notes_df.iterrows(),
                                                                 total=len(notes_df),
                                                                 # max_workers=16
                                                                 )

    return doc_results_list

def cardio_oncology_pipeline(notes_df):
    project_dir = "/Workspace/Users/vamarvan23@osfhealthcare.org/"
    # remote dir shared
    current_datetime_str = datetime.datetime.now().strftime("%Y%m%d")
    out_dir = f"/Workspace/Shared/NLP/{current_datetime_str}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    rule_set = get_ruleset()

    doc_results_list = run_medspacy(notes_df, rule_set, "NOTE_TEXT")

    export_as_label_studio_format(doc_results_list, out_dir)


def main():
    notes_df = get_notes_df(spark)

    cardio_oncology_pipeline(notes_df)


if __name__ == "__main__":
    cardio_oncolgy_pipeline()

