import numpy as np
import pandas as pd
import csv

cervical_results = pd.read_csv("artifacts/results/cervical_notes.csv")

cervical_results.rename(columns={"context": "long context"}, inplace=True)
cervical_notes_with_labels = pd.read_excel("datasets/cervical/data_with_phi/cervical_notes_HKH_labeled.xlsx")

cervical_notes_with_labels = cervical_notes_with_labels.dropna(subset=["GOLD STANDARD"])

columns_to_filter = ["long context", "matched_pattern", "label", "masked_ids",
                     "DX_CODE", "PROC_CODE",
                     "HPV_TEST_CODE",
                     "HPV_TEST_RESULT"
                     ]

columns = ["prediction_id",
           "matched_pattern",
           "pred_label",
           "gold_label",
           "additional_info"]
labels_list = []

# to string
cervical_notes_with_labels["PROC_CODE"] = cervical_notes_with_labels["PROC_CODE"].apply(lambda x: str(x))
cervical_notes_with_labels["masked_ids"] = cervical_notes_with_labels["masked_ids"].apply(lambda x: str(x))
labels_with_multiple_results = []
not_found_rows = []

for i, row in cervical_notes_with_labels.iterrows():
    result_slice = cervical_results
    for f in columns_to_filter:
        if len(result_slice) == 0:
            break
        column_value = row[f]
        if f in ["masked_ids"]:
            column_value = int(column_value)

        if pd.notna(column_value):
            pre_slice = result_slice
            result_slice = result_slice[result_slice[f] == column_value]
            if len(result_slice) == 0:
                print(f"No results found for row: {i}")
                not_found_rows.append(row)
    if len(result_slice) > 1:
        print("Multiple results found for row: ", i)

    for j, r in result_slice.iterrows():
        prediction_id = r["prediction_id"]
        matched_pattern = r["matched_pattern"]
        pred_label = r["label"]
        gold_label = row["GOLD STANDARD"]
        additional_info = row["Explanation"]
        labels_list.append([prediction_id, matched_pattern, pred_label, gold_label, additional_info])

labels_df = pd.DataFrame(labels_list, columns=columns)

not_found_df = pd.DataFrame(not_found_rows)

labels_df.to_csv("artifacts/results/cervical_labels.csv", index=False)
not_found_df.to_csv("artifacts/results/cervical_labels_not_found.csv", index=False)
