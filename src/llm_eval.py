import json
import re

import requests
import pandas as pd
from tqdm import tqdm
import data_loader as data_loader
import medspacy_pipeline
import regex as re
# client = Client(url)
# text = ""
# for response in client.generate_stream("What is Deep Learning?", max_new_tokens=20):
#     if not response.token.special:
#         text += response.token.text
# print(text)

def get_llm_response(prompt_text):
    url = 'http://192.168.1.100:3000/generate'
    data = {
        'inputs': prompt_text,
        'parameters': {'max_new_tokens': 30}
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        result = response.json()
        generated_text = result['generated_text']
    else:
        generated_text = f"Error: {response.status_code} - {response.text}"
        print(generated_text)

    return generated_text, response.status_code == 200


def run_llm_prompting(medspacy_doc_dict):
    with open("src/prompt.txt", "r") as f:
        prompt_text = f.read()

    prediction_list = []

    total_ent_count = sum([len(doc.ents) for doc in medspacy_doc_dict.values()])

    total_ent_count = min(total_ent_count, 500)
    progress_bar = tqdm(total=total_ent_count)

    for file, medspacy_doc in medspacy_doc_dict.items():
        sentence_list = list(medspacy_doc.sents)

        for i, sentence in enumerate(sentence_list):
            for ent in sentence.ents:
                previous_sentence = ""
                if i > 0:
                    previous_sentence = sentence_list[i - 1].text
                next_sentence = ""
                if i < len(sentence_list) - 1:
                    next_sentence = sentence_list[i + 1].text

                current_sentence = sentence.text

                query = f'{prompt_text}' \
                        f'Context: "{previous_sentence + current_sentence}"' \
                        f'Span: "{ent.text}"' \
                        f'Assertion: '

                response, success = get_llm_response(query)

                # print(f"current sentence: {current_sentence}"
                #       f"entity: {ent.text}"
                #       f"response: {response}")
                prediction = (file, ent.start_char, ent.end_char, response)
                prediction_list.append(prediction)
                progress_bar.update(1)
                progress_bar.refresh()

    prediction_df = pd.DataFrame(prediction_list, columns=['file', 'start_char', 'end_char', 'label_pred'])

    prediction_df.to_csv('src/predictions.csv')
    return prediction_df


def parse_label_pred(llm_output):
    word_list = llm_output.split()

    label = None
    if len(word_list) > 0:
        if "error" not in llm_output.lower():
            label = word_list[0].strip()
            label = re.sub(r'[^\w\s]', '', label)

    return label



def main():
    llm_results = pd.read_csv('artifacts/results/Arvan_etal/llama_2_1.csv')

    llm_results["label_pred"] = llm_results["label_pred"].apply(parse_label_pred)
    prediction_df = llm_results[llm_results.columns[1:]]

    prediction_list = list(prediction_df.itertuples(index=False, name=None))
    labels_list_out_path = "datasets/i2b2/data/assertion.csv"
    labels_list = list(pd.read_csv(labels_list_out_path, index_col=0).itertuples(index=False, name=None))

    # annotation_list, dataset_text_dict = data_loader.I2B2_2010_Train.read_i2b2_data()

    medspacy_pipeline.evaluate(prediction_list, labels_list, "llama_2")

    print("")
    # evaluate(predictions, labels_list)


if __name__ == '__main__':
    main()
