import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

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
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
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

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    meter = BenchmarkMeter()
    meter.start()

    step = 0
    for batch in loader:
        if step >= args.max_steps:
            break

        batch = {k: v.to(model_engine.device) for k, v in batch.items()}

        outputs = model_engine(**batch)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

        meter.update(batch["input_ids"])

        if model_engine.global_rank == 0 and step % 5 == 0:
            print(f"[deepspeed] step={step:03d} loss={loss.item():.4f}")

        step += 1

    if model_engine.global_rank == 0:
        metrics = meter.summary()
        print_summary(f"DeepSpeed {args.deepspeed_config}", metrics)


if __name__ == "__main__":
    main()
