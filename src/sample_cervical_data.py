import argparse
import logging
import time

import pandas as pd

from src import medspacy_pipeline_cervical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Running {__file__}")


def main():
    start_timer = time.perf_counter()
    dataset_dict, notes_column = medspacy_pipeline_cervical.load_cervical_data()

    notes_count = 100

    # sampling the count from the train set
    sampled_notes_df = dataset_dict["train"]["notes"].sample(n=notes_count)

    # reorder the columns, moving note_id to the front
    notes_columns = sampled_notes_df.columns.tolist()
    notes_columns.remove("note_id")
    notes_columns = ["note_id"] + notes_columns
    sampled_notes_df = sampled_notes_df[notes_columns]

    # covert to list of dicts
    selected_notes = sampled_notes_df.to_dict(orient="records")

    medspacy_pipeline_cervical.export_as_label_studio_format(selected_notes,
                                                             "artifacts/results/",
                                                             file_name="cervical-annotation-3.json"
                                                             remove_predictions=True)

    out_dir = "artifacts/results/"
    file_name = "cervical-annotation-3.json"

    out_file_path = f"{out_dir}/{file_name}"


    with open(out_file_path, "w") as f:
        json.dump(selected_notes, f, indent=2)

    # export_as_label_studio_format(selected_notes, "artifacts/results/", remove_predictions=True)

    logger.info("Evaluated in {:.2f} seconds".format(time.perf_counter() - start_timer))


if __name__ == '__main__':
    main()
    logger.info("Done")
