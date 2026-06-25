import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import TokenizedTextDataset, make_collate_fn
from utils import BenchmarkMeter, print_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.train()

    dataset = TokenizedTextDataset(
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    meter = BenchmarkMeter()
    meter.start()

    step = 0
    for batch in loader:
        if step >= args.max_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        meter.update(batch["input_ids"])

        if step % 5 == 0:
            print(f"[baseline] step={step:03d} loss={loss.item():.4f}")

        step += 1

    metrics = meter.summary()
    print_summary("Baseline PyTorch", metrics)


if __name__ == "__main__":
    main()
