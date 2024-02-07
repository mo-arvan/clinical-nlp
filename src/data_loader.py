import os
import re

import medspacy
import pandas as pd
from spacy.tokens import Span
from tqdm import tqdm


class I2B2_2010_Train:
    # Define a function to parse each line of the AST file
    @staticmethod
    def parse_ast_file(file_path):
        """
        c=”concept text” offset_line_start:offset_word_start offset_line_end:offset_word_end ||t=”concept type” ||
        a=”assertion value”
        Examples:
        c=”prostate cancer” 5:7 5:8||t=”problem”||a=”present”
        c=”diabetes” 2:14 2:14||t=”problem”||a=”absent”
        c=”pain” 7:3 7:3||t=”problem”||a=”conditional”
        :param file_path:
        :return:
        """
        base_file_name = os.path.basename(file_path)[:-4]
        # Define a regular expression pattern to extract the relevant information
        pattern = r'c="([^"]+)" (\d+):(\d+) (\d+):(\d+)\|\|t="([^"]+)"\|\|a="([^"]+)"'

        # Create a list to store parsed data
        parsed_data = []

        # Read the AST file line by line and parse each line
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                # Use re.search to find the matches in the line
                match = re.search(pattern, line)
                if match:
                    concept_text = match.group(1)
                    line_start = int(match.group(2))
                    word_start = int(match.group(3))
                    line_end = int(match.group(4))
                    word_end = int(match.group(5))
                    concept_type = match.group(6)
                    assertion_value = match.group(7)

                    parsed_data.append({
                        'File Name': base_file_name,
                        'Line Number Start': line_start,
                        'Line Number End': line_end,
                        'Word Number Start': word_start,
                        'Word Number End': word_end,
                        'Concept Text': concept_text,
                        'Concept Type': concept_type,
                        'Assertion Value': assertion_value
                    })

        # remove duplicates
        current_len = len(parsed_data)
        parsed_data = [dict(t) for t in {tuple(d.items()) for d in parsed_data}]
        if current_len != len(parsed_data):
            print(f"Warning: removed {current_len - len(parsed_data)} duplicates from {file_path}")

        return parsed_data

    @staticmethod
    def read_i2b2_data():
        dataset_path = "datasets/i2b2/data/concept_assertion_relation_training_data"

        datasets_sub_dir = ["beth", "partners"]

        annotation_list = []
        dataset_text_dict = {}
        for sub_dir in datasets_sub_dir:
            assert_dir = os.path.join(dataset_path, sub_dir, "ast")
            text_dir = os.path.join(dataset_path, sub_dir, "txt")
            # Read all the AST files in the assert directory
            for ast_file in os.listdir(assert_dir):
                if ast_file.endswith(".ast"):
                    file_path = os.path.join(assert_dir, ast_file)
                    annotation_list.append(I2B2_2010_Train.parse_ast_file(file_path))
                    # print(file)
                    # print(os.path.join(assert_dir, file))
            for text_file in os.listdir(text_dir):
                if text_file.endswith(".txt"):
                    file_path = os.path.join(text_dir, text_file)
                    file_base_name = os.path.basename(file_path)[:-4]
                    with open(file_path, 'r') as f:
                        dataset_text_dict[file_base_name] = f.read()

        annotation_list = [item for sublist in annotation_list for item in sublist]

        return annotation_list, dataset_text_dict

    @staticmethod
    def ensure_dataset_correctness(dataset_df):
        assert_label_count_dict = dataset_df["Assertion Value"].value_counts().to_dict()

        wang_et_al_stats = pd.read_csv("artifacts/results/Wang_etal/table_2.csv")
        wang_melt = wang_et_al_stats[wang_et_al_stats["Dataset"] == "i2b2 2010 Train"].melt(id_vars=["Dataset"],
                                                                                            var_name='label',
                                                                                            value_name='count')
        wang_melt["count"] = wang_melt["count"].astype(int)
        i2b2_stats = dict(wang_melt[["label", "count"]].itertuples(index=False))
        # 'Present', 'Absent', 'Possible', 'Hypothetical', 'Conditional', 'Not Associated']
        # ['present', 'absent', 'possible', 'hypothetical', 'conditional' 'associated_with_someone_else', ]

        label_map = {"present": "Present",
                     "absent": "Absent",
                     "possible": "Possible",
                     "hypothetical": "Hypothetical",
                     "conditional": "Conditional",
                     "associated_with_someone_else": "Not Associated"}
        # Compare the assertion value counts to the Wang et al. paper
        for label in label_map:
            assert label in assert_label_count_dict
            assert assert_label_count_dict[label] == i2b2_stats[label_map[label]]


def load_i2b2_2010_train():
    # Flatten the list of lists
    annotation_list, dataset_text_dict = I2B2_2010_Train.read_i2b2_data()
    # dataset_df = pd.DataFrame(annotation_list)

    # I2B2_2010_Train.ensure_dataset_correctness(dataset_df)

    # Convert the list of dictionaries to a pandas DataFrame

    # nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_context"])
    nlp = medspacy.load(medspacy_enable=["medspacy_pyrush"])
    # context = nlp.get_pipe("medspacy_context")

    # file_map_dict = {}
    # for file_name, doc in dataset_text_dict.items():
    #     file_lines = doc.split("\n")
    #     char_index = 0
    #     line_word_index_map = {}
    #     for i, line in enumerate(file_lines):
    #         word_index_map = {}
    #         for j, word in enumerate(line.split()):
    #             word_index_map[j] = char_index
    #             char_index += len(word) + 1
    #         line_word_index_map[i] = word_index_map
    #     file_map_dict[file_name] = line_word_index_map
    medspacy_doc_dict = {}
    labels_list = []

    for file_name, text in tqdm(dataset_text_dict.items()):
        current_file_annotation_list = [a for a in annotation_list if a["File Name"] == file_name]

        text_normalized = text.lower()
        new_line_indices = [0] + [m.regs[0][0] for m in re.finditer(r'\n', text_normalized)] + [len(text_normalized)]

        medspacy_doc_dict[file_name] = nlp(text)
        spacy_doc = medspacy_doc_dict[file_name]

        entity_list = []

        for i, row in enumerate(current_file_annotation_list):
            file_name = row["File Name"]
            line_number_start = row["Line Number Start"]
            line_number_end = row["Line Number End"]
            word_number_start = row["Word Number Start"]
            word_number_end = row["Word Number End"]
            annotation_text = row["Concept Text"]

            concept_type = row["Concept Type"]
            assertion_value = row["Assertion Value"]

            # FROM DATASET DOC file 'Annotation File Formatting.pdf'
            # offset represents the beginning and end line and word numbers that span the
            # concept text. An offset is formatted as the line number followed by a colon
            # followed by the word number. The starting and ending offset are separated
            # by a space. The first line of a report starts as line number 1. The first word in
            # a line is counted as word number 0.

            # the offsets are numbers, line number is 1 indexed, word number is 0 indexed
            word_start_index = word_number_start
            word_end_index = word_number_end + 1
            line_start_index = line_number_start - 1
            line_end_index = line_number_end

            search_window_start_index = new_line_indices[line_start_index] + 1
            search_window_end_index = new_line_indices[line_end_index]
            search_window_text = text_normalized[search_window_start_index: search_window_end_index]
            search_window_tokens_start_list = [0] + [m.regs[0][0] + 1 for m in re.finditer(r' ', search_window_text)] + \
                                              [len(search_window_text) + 1]

            if word_end_index > len(search_window_tokens_start_list):
                raise ValueError(f"Word end is greater than the length of the search window in {file_name}, "
                                 f"search window: {search_window_text}")

            annotation_span = search_window_tokens_start_list[word_start_index], search_window_tokens_start_list[
                                                                                     word_end_index] - 1
            annotation_text_matched = search_window_text[annotation_span[0]: annotation_span[1]]

            if annotation_text_matched == annotation_text:
                span_start = search_window_start_index + annotation_span[0]
                span_end = search_window_start_index + annotation_span[0] + len(annotation_text)
            else:
                # Fall back to regex search, not ideal
                # print(
                #     f"Fall back to regex search for {file_name}"
                #     f"annotation text matched: {annotation_text_matched}"
                #     f"annotation text: {annotation_text}"
                #     # f"search window: {search_window_text}"
                # )
                re_matches = list(re.finditer(re.escape(annotation_text), search_window_text))

                if len(re_matches) == 0:
                    raise ValueError(
                        f"No matches found for {annotation_text} in {file_name}, search window: {search_window_text}"
                    )
                elif len(re_matches) == 1:
                    span_start = re_matches[0].regs[0][0] + search_window_start_index
                    span_end = re_matches[0].regs[0][1] + search_window_start_index
                else:
                    matches_distance = [(abs(r.regs[0][0] - search_window_tokens_start_list[word_number_start]), r)
                                        for r in re_matches]

                    closets_distance, closet_match = min(matches_distance, key=lambda x: x[0])
                    if closets_distance > 0:
                        print(
                            f"Closest match found for {annotation_text} in {file_name}, search window: {search_window_text}")

                    span_start = closet_match.regs[0][0] + search_window_start_index + 1
                    span_end = closet_match.regs[0][1] + search_window_start_index + 1

                    if text_normalized[span_start:span_end].lower() != annotation_text:
                        raise ValueError(
                            f"Match found for {annotation_text} in {file_name}, search window: {search_window_text}, but the match was not the same as the annotation text"
                        )

            new_ent = spacy_doc.char_span(span_start, span_end, label=assertion_value)
            label = (file_name, span_start, span_end, assertion_value)
            labels_list.append(label)

            if annotation_text is None or new_ent is None or new_ent.text.lower() != annotation_text.lower():
                spacy_doc.char_span(span_start, span_end, label=assertion_value)
                print(
                    f"Match found for {annotation_text} in {file_name}, search window: {search_window_text},"
                    f" but the match was not the same as the annotation text")
            #     now we set the span as Span in the doc
            entity_list.append(new_ent)
        try:
            spacy_doc.ents = tuple(entity_list)
        except Exception as e:
            print(f"Error setting entities for {file_name}")

    labels_list_out_path = "datasets/i2b2/data/assertion.csv"
    if not os.path.isfile(labels_list_out_path):
        labels_df = pd.DataFrame(labels_list, columns=['file', 'start_char', 'end_char', 'label'])
        labels_df.to_csv(labels_list_out_path)

    return medspacy_doc_dict, labels_list


def load_i2b2_2010_test():
    pass


def load_mimic_iii():
    pass


def load_neg_corpus():
    pass
