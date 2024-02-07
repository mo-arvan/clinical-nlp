import medspacy
import numpy as np
import pandas as pd
from medspacy.ner import TargetRule
from tqdm import tqdm
import re

def load_terms(term_csv_file):
    nlp_terms = pd.read_csv(term_csv_file)
    terms_list = []

    for i, row in nlp_terms.iterrows():
        label = row["Discrete Data Element"]

        for jj in range(1, len(row)):
            if row[jj] is not np.nan:
                terms_list.append((row[jj], label))

    return terms_list


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
    nlp_dir = "/Workspace/Users/vamarvan23@osfhealthcare.org/nlp/"

    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [TargetRule(literal=r[0].strip(), category=r[1]) for r in labels_list]
    target_matcher.add(target_rules)

    result_list = []
    matched_pattern_list = []

    for i, row in tqdm(notes_df.iterrows(), total=len(notes_df)):
        doc = nlp(row["NOTE_TEXT"])
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
                    value = ""

                    if matched_pattern == "left ventricular ejection fraction":
                        # Define regex patterns
                        pattern1 = re.compile(r'Left ventricular ejection fraction (\d{1,2}-\d{1,2})%')
                        pattern2 = re.compile(r'left ventricular ejection fraction of (\d{1,2})%')
                        pattern3 = re.compile(r'left ventricular ejection fraction is (\d{1,2})%')

                        # Extract values using regex patterns
                        match1 = pattern1.search(sentence_text)
                        match2 = pattern2.search(sentence_text)
                        match3 = pattern3.search(sentence_text)

                        if match1:
                            value = match1.group(1)
                        elif match2:
                            value = match2.group(1)
                        elif match3:
                            value = match3.group(1)
                        else:
                            print(f"failed to extract value for {matched_pattern} in {sentence_text}")
                    elif matched_pattern == "LVEF":
                        pattern1 = re.compile(r'LVEF is (\d{1,2})%')
                        pattern2 = re.compile(r'LVEF (\d{1,2}-\d{1,2})%')
                        pattern3 = re.compile(r'LVEF of (\d{1,2}-\d{1,2})%')

                        match1 = pattern1.search(sentence_text)
                        match2 = pattern2.search(sentence_text)
                        match3 = pattern3.search(sentence_text)

                        if match1:
                            value = match1.group(1)
                        elif match2:
                            value = match2.group(1)
                        elif match3:
                            value = match3.group(1)
                        else:
                            print(f"failed to extract value for {matched_pattern} in {sentence_text}")

                    results_dict = {
                        "10 words context": window_context_10,
                        "20 words context": window_context_20,
                        "sentence": sentence_text,
                        "matched_pattern": matched_pattern,
                        "label": label,
                        "assertion": assertion,
                        "value": value,
                        **row[:-1].to_dict()
                    }
                    result_list.append(results_dict)

        if (i + 1) % 10000 == 0:
            results_df = pd.DataFrame(result_list)
            batch_number = (i + 1) // 10000
            results_df.to_csv(nlp_dir + f"cardio_oncology_batch_{batch_number}.csv", index=False)
            result_list = []
