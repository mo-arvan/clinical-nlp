import medspacy
import numpy as np
import pandas as pd
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent
from tqdm import tqdm
import pickle
import os
import data_loader as data_loader
import radtext


def read_data():
    medspacy_doc_dict, labels_list = data_loader.load_i2b2_2010_train()

    return medspacy_doc_dict, labels_list
    #
    # notes = pd.read_csv(notes_file_path)["NoteText"].tolist()
    # rules_df = pd.read_csv(rules_file_path)
    #
    # rules_list = []
    # for i, row in rules_df.iterrows():
    #     rule_name = row[0]
    #
    #     rule_patterns = [{"LOWER": pp} for p in row[1:] if isinstance(p, str) for pp in p.split()]
    #     if len(rule_patterns) > 0:
    #         rules_list.append((rule_name, rule_patterns))

    # rules_list = [rules_list[0]]
    #
    # notes = [notes[48]]
    #
    # print(notes[0])

    # def exclude_range_or_invalid(x):
    #     if "=" in x:
    #         return False
    #     elif ">" in x:
    #         return False
    #     elif "<" in x:
    #         return False
    #     elif " to " in x:
    #         return False
    #     elif "EF 51%-54%" == x:
    #         return False
    #     elif 'Elevated CPK (LOINC) - need a value range (???)' == x:
    #         return False
    #     elif '(' in x:
    #         return False
    #     elif ')' in x:
    #         return False
    #     elif "and" in x:
    #         return False
    #     else:
    #         return True

    # rules_list = list(filter(exclude_range_or_invalid, rules_list))


def run_target_matcher(notes):
    # Add rules for target concept extraction
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    # target_rules = [
    #     TargetRule("atrial fibrillation", "PROBLEM"),
    #     TargetRule("atrial fibrillation", "PROBLEM", pattern=[{"LOWER": "afib"}]),
    #     TargetRule("pneumonia", "PROBLEM"),
    #     TargetRule("Type II Diabetes Mellitus", "PROBLEM",
    #                pattern=[
    #                    {"LOWER": "type"},
    #                    {"LOWER": {"IN": ["2", "ii", "two"]}},
    #                    {"LOWER": {"IN": ["dm", "diabetes"]}},
    #                    {"LOWER": "mellitus", "OP": "?"}
    #                ]),
    #     TargetRule("warfarin", "MEDICATION")
    # ]

    # target_rules = [TargetRule(literal=r[0], category="PROBLEM", pattern=r[1]) for r in rules_list]

    target_rules = [TargetRule(literal=r[0], category="PROBLEM", pattern=r[1]) for r in rules_list]

    target_matcher.add(target_rules)

    number_list = []
    for i, text_row in tqdm(enumerate(notes)):

        # print(f"Row {i}")
        # print(text_row)
        doc = nlp(text_row)
        number_of_entities = len(doc.ents)
        number_list.append(number_of_entities)
        if number_of_entities >= 1:
            print(i, set([ent.label_ for ent in doc.ents]))
            doc.ents = [ent for ent in doc.ents if ent.label_ == "PROBLEM"]
            html = visualize_ent(doc, jupyter=False)
            # Save to file
            with open(f"out/{number_of_entities}_row_{i}.html", "w") as f:
                f.write(html)

        # print("\n\n")

    print(np.mean(number_list))
    print(np.std(number_list))
    print(np.max(number_list))
    print(np.min(number_list))
    print(np.median(number_list))


def run_radtex_context(medspacy_doc_dict):
    nlp = radtext.Pipeline()
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


def evaluate(predictions, labels_list, model_name):
    predictions = sorted(predictions, key=lambda x: (x[0], x[1], x[2]))
    labels_list = sorted(labels_list, key=lambda x: (x[0], x[1], x[2]))

    predictions_df = pd.DataFrame(predictions, columns=["file_name", "start", "end", "label"])
    labels_df = pd.DataFrame(labels_list, columns=["file_name", "start", "end", "label"])

    # rename label column to label_gold
    labels_df = labels_df.rename(columns={"label": "label_gold"})
    # rename label column to label_pred
    predictions_df = predictions_df.rename(columns={"label": "label_pred"})

    merged_df = pd.merge(labels_df, predictions_df, how="left", on=["file_name", "start", "end"])

    # ensure there not no duplicate matches
    match_count = merged_df[["file_name", "start", "end"]].groupby(by=["file_name", "start", "end"]).size().reset_index(
        name="count")
    if sum(match_count["count"] > 1) > 0:
        filtered_df = match_count.loc[match_count["count"] > 1]
        print(f"Found {len(filtered_df)} duplicate matches")

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

    column_order = ["label", "present", "absent", "possible", "conditional", "hypothetical",
                    "associated_with_someone_else",
                    "micro", "macro"]
    result_df = result_df[column_order]

    result_df = result_df.round(3)

    result_df.to_csv(f"artifacts/results/{model_name}.csv")
    result_df.to_latex(f"artifacts/results/{model_name}.tex")

    print(result_df)


def main():
    cache_file = ".cache/medspacy_doc_dict.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            medspacy_doc_dict, labels_list = pickle.load(f)
    else:
        medspacy_doc_dict, labels_list = read_data()
        with open(cache_file, "wb") as f:
            pickle.dump((medspacy_doc_dict, labels_list), f)

    predictions = run_radtex_context(medspacy_doc_dict)
    evaluate(predictions, labels_list, "medspacy_context")


if __name__ == '__main__':
    main()
