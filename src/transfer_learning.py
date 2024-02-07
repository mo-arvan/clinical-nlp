import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments


# Load and preprocess the data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cue_spans = self.data['cue_span']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.loc[idx, 'sentence']
        cue_span = self.cue_spans[idx]

        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length)

        # Prepare target labels for token classification
        labels = [-100] * self.max_length  # -100 is used to ignore non-entity tokens
        if not pd.isna(cue_span):
            cue_span = eval(cue_span)  # Convert string representation to a list
            for cue in cue_span:
                labels[cue[0]:cue[1] + 1] = [1] * (cue[1] - cue[0] + 1)

        encoding['labels'] = labels
        encoding = {key: torch.tensor(val) for key, val in encoding.items()}

        return encoding


def fine_tune_model(data_path, model_name='bert-base-uncased', batch_size=16, max_length=128, num_epochs=3):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=2)

    # Prepare the dataset
    dataset = CustomDataset(data_path, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=5000,
        save_total_limit=2,
        logging_steps=100,
        logging_dir="./logs",
        logging_first_step=True,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


# Fine-tune the model
data_path = 'datasets/bioscope_corpus/bioscope_abstract.csv'
fine_tune_model(data_path)
