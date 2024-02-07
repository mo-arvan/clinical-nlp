import medspacy
import numpy as np
import pandas as pd
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent
from tqdm import tqdm
import spacy
import data_loader as data_loader
from collections import Counter


def load_terms(term_csv_file):
    nlp_terms = pd.read_csv(term_csv_file)
    terms_list = []

    for i, row in nlp_terms.iterrows():
        label = row["Discrete Data Element"]

        for jj in range(1, len(row)):
            if row[jj] is not np.nan:
                terms_list.append((row[jj], label))

    return terms_list


def read_data():
    notes = pd.read_csv("datasets/cardio-oncology/data/notes.csv")
    terms_list = load_terms("datasets/cardio-oncology/data/terms.csv")

    return notes, terms_list


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


def run_medspacy(notes_df, labels_list):
    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [TargetRule(literal=r[0].strip(), category=r[1]) for r in labels_list]
    target_matcher.add(target_rules)

    result_list = []
    matched_pattern_list = []

    for i, row in tqdm(notes_df.iterrows(), total=len(notes_df)):
        doc = nlp(row["NoteText"])

        if row["NOTE_ID"] == 3871638323:
            print("the one with obesity")
        if len(doc.ents) > 0:

            for sentence in doc.sents:
                for ent in sentence.ents:
                    sentence_text = sentence.text
                    matched_pattern = ent.text
                    label = ent.label_
                    assertion = get_medspacy_label(ent)

                    window_context_10 = " ".join([token.text for token in sentence.doc[ent.start - 10:ent.end + 10]])
                    window_context_20 = " ".join([token.text for token in sentence.doc[ent.start - 20:ent.end + 20]])

                    matched_pattern_list.append(matched_pattern.lower().strip())
                    results_dict = {
                        "10 words context": window_context_10,
                        "20 words context": window_context_20,
                        "sentence": sentence_text,
                        "matched_pattern": matched_pattern,
                        "label": label,
                        "assertion": assertion,
                        **row[:-1].to_dict()
                    }
                    result_list.append(results_dict)

    results_df = pd.DataFrame(result_list)

    results_df.to_csv("artifacts/results/cardio_oncology_notes.csv", index=False)

    matched_pattern_counter = Counter(matched_pattern_list)

    for literal, label in labels_list:
        l = literal.lower().strip()
        if l not in matched_pattern_counter:
            matched_pattern_counter[l] = 0

    matched_pattern_df = pd.DataFrame.from_dict(matched_pattern_counter, orient='index').reset_index()
    matched_pattern_df = matched_pattern_df.rename(columns={"index": "matched_pattern", 0: "count"})
    matched_pattern_df = matched_pattern_df.sort_values(by="count", ascending=False)
    matched_pattern_df.to_csv("artifacts/results/matched_patterns.csv", index=False)

    raise Exception("Done")
    context = nlp.get_pipe("medspacy_context")

    # i2b2 labels
    # 1. present
    # 2. absent
    # 3. possible
    # 4. conditional
    # 5. hypothetical
    # 6. associated_with_someone_else

    context_to_i2b2_label_map = {
        "NEGATED_EXISTENCE": "absent",
        'POSSIBLE_EXISTENCE': "possible",
        "CONDITIONAL_EXISTENCE": "conditional",
        "HYPOTHETICAL": "hypothetical",
        'HISTORICAL': "historical",
        'FAMILY': "associated_with_someone_else"
    }

    predicted_list = []
    for file_name, doc in medspacy_doc_dict.items():
        context(doc)

        # for target, modifier in doc._.context_graph.edges:
        #     prediction = (file_name, target.start_char, target.end_char, modifier.category)
        #     predicted_list.append(prediction)
        for ent in doc.ents:
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
                    label = "not_associated"

            prediction = (file_name, ent.start_char, ent.end_char, label)
            predicted_list.append(prediction)
    print(f"Loaded model with pipeline components: {[pipe for pipe in nlp.pipe_names]}")

    return predicted_list
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
    notes_df, labels_list = read_data()
    predictions = run_medspacy(notes_df, labels_list)
    evaluate(predictions, labels_list)


if __name__ == '__main__':
    main()
