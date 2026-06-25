import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class TokenizedTextDataset(Dataset):
    def __init__(self, tokenizer, block_size=128, max_samples=512):
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [x["text"] for x in raw if x["text"].strip()]
        texts = texts[:max_samples]

        joined = "\n\n".join(texts)
        tokenized = tokenizer(joined, return_tensors="pt", truncation=False)
        ids = tokenized["input_ids"][0]

        self.examples = []
        for i in range(0, len(ids) - block_size, block_size):
            chunk = ids[i : i + block_size]
            self.examples.append(chunk)

        if len(self.examples) == 0:
            raise ValueError("No examples created. Try smaller block_size.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def make_collate_fn(tokenizer):
    def collate(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    return collate
