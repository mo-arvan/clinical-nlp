# ! pip install -q numpy medspacy tqdm spacy
# dbutils.library.restartPython()

import concurrent.futures
import datetime
import functools
import os
import re

import medspacy
import numpy as np
import pandas as pd
from medspacy.ner import TargetRule
from tqdm import tqdm


def load_terms(term_csv_file):
    nlp_terms = pd.read_csv(term_csv_file)
    terms_list = []

    for i, row in nlp_terms.iterrows():
        label = row["Discrete Data Element"]

        for jj in range(1, len(row)):
            if row[jj] is not np.nan:
                terms_list.append((row[jj], label))

    return terms_list


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


def parse_value_from_sentence(sentence_text, label, matched_pattern):
    value = None
    if label in ["Global Longitudinal Strain",
                 "EF 51%-54%", ]:
        value_patterns = [
            re.compile(r'(\d{1,2}-\d{1,2})%'),
            re.compile(r'(\d{1,2})%'),
        ]
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
    # row_dict.pop(note_text_column, None)
    results_list = []
    if len(doc.ents) > 0:
        for sentence in doc.sents:
            for ent in sentence.ents:
                sentence_text = sentence.text
                matched_pattern = ent.text
                label = ent.label_
                assertion = get_medspacy_label(ent)
                value = parse_value_from_sentence(sentence_text, label, matched_pattern)

                ent_20_window_start = max(ent.start - 10, 0)
                ent_20_window_end = min(ent.end + 10, len(sentence.doc))
                window_context_20 = " ".join(
                    [token.text for token in sentence.doc[ent_20_window_start:ent_20_window_end]])

                prediction_id = f"note_{note_id}_ent_{ent.start}_{ent.end}"

                results_dict = {
                    "context": window_context_20,
                    "matched_pattern": matched_pattern,
                    "label": label,
                    "assertion": assertion,
                    "prediction_id": prediction_id,
                    "value": value,
                    **row_dict
                }
                results_list.append(results_dict)

    return results_list


def get_notes_df(spark, ids_list):
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


def run_medspacy(notes_df, labels_list, out_dir):
    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush",
                                         "medspacy_target_matcher",
                                         "medspacy_context"])

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [TargetRule(literal=r[0].strip(), category=r[1]) for r in labels_list]
    target_matcher.add(target_rules)

    notes_count = 0
    notes_df["note_id"] = notes_df.index + notes_count

    run_medspacy_on_note_partial = functools.partial(run_medspacy_on_note,
                                                     medspacy_nlp=nlp,
                                                     note_text_column="NOTE_TEXT")
    doc_list = run_in_parallel_cpu_bound(run_medspacy_on_note_partial,
                                         notes_df.iterrows(),
                                         total=len(notes_df),
                                         max_workers=10)

    result_list = [result for doc in doc_list for result in doc if len(doc) > 0]

    results_df = pd.DataFrame(result_list)

    out_file = f"{out_dir}/results.csv"
    results_df.to_csv(out_file, index=False)


project_dir = "/Workspace/Users/vamarvan23@osfhealthcare.org/"
# remote dir shared
current_datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/Workspace/Shared/NLP/{current_datetime_str}"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

nlp_terms = load_terms(f"{project_dir}/nlp_terms.csv")
ids_list = get_ids_of_interest(spark)
notes_df = get_notes_df(spark, ids_list)


run_medspacy(notes_df, nlp_terms, out_dir)
