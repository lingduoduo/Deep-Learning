import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    def __init__(self, linear, scales, quantized_weight, protected_mask=None, fp_weight=None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.register_buffer("scales", scales)
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer(
            "protected_mask",
            protected_mask if protected_mask is not None else torch.zeros(linear.out_features, dtype=torch.bool),
        )
        self.register_buffer(
            "fp_weight",
            fp_weight if fp_weight is not None else torch.zeros_like(linear.weight.detach()),
        )

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        dequantized_weight = self.quantized_weight.float() * self.scales
        if self.protected_mask.any():
            dequantized_weight[self.protected_mask] = self.fp_weight[self.protected_mask]
        return F.linear(x, dequantized_weight, self.bias)


class AWQQuantizer:
    def __init__(self, w_bit=4, group_size=128, protect_ratio=0.01):
        self.w_bit = w_bit
        self.group_size = group_size
        self.protect_ratio = protect_ratio

    def quantize_model(self, model, calibration_data):
        """Main AWQ quantization workflow."""
        quantized_model = copy.deepcopy(model)

        # 1) Collect activation statistics.
        activation_stats = self._collect_activation_stats(
            quantized_model,
            calibration_data,
        )

        # 2) Compute per-layer importance scores.
        importance_scores = self._calculate_importance_scores(activation_stats)

        # 3) Select weight channels to protect.
        protected_channels = self._select_protected_channels(importance_scores)

        # 4) Apply quantization.
        self._apply_quantization(quantized_model, protected_channels)

        return quantized_model, protected_channels

    def _collect_activation_stats(self, model, calibration_data):
        """Collect activation statistics from linear layers."""
        stats = {}

        def hook_fn(name):
            def hook(module, inputs, output):
                activations = output.detach()
                if activations.dim() > 2:
                    activations = activations.reshape(-1, activations.size(-1))

                if name not in stats:
                    stats[name] = {
                        "abs_sum": torch.zeros(activations.size(-1)),
                        "abs_max": torch.zeros(activations.size(-1)),
                        "count": 0,
                    }

                layer_stats = stats[name]
                layer_stats["abs_sum"] += activations.abs().sum(dim=0).cpu()
                layer_stats["abs_max"] = torch.maximum(
                    layer_stats["abs_max"],
                    activations.abs().max(dim=0).values.cpu(),
                )
                layer_stats["count"] += activations.size(0)

            return hook

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)

        for hook in hooks:
            hook.remove()

        if was_training:
            model.train()

        return stats

    def _calculate_importance_scores(self, activation_stats):
        """Compute importance scores for output channels."""
        importance_scores = {}

        for layer_name, stats in activation_stats.items():
            mean_abs = stats["abs_sum"] / max(stats["count"], 1)
            max_abs = stats["abs_max"]
            importance = 0.7 * mean_abs + 0.3 * max_abs
            importance_scores[layer_name] = importance

        return importance_scores

    def _select_protected_channels(self, importance_scores):
        """Select important output channels to keep at higher precision."""
        protected_channels = {}

        for layer_name, scores in importance_scores.items():
            num_channels = scores.numel()
            num_protect = max(1, int(num_channels * self.protect_ratio))
            _, top_indices = torch.topk(scores, num_protect)
            protected_channels[layer_name] = top_indices

        return protected_channels

    def _apply_quantization(self, model, protected_channels):
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            scales, quantized_weight, protected_mask, fp_weight = self._quantize_linear_weight(
                module.weight.detach(),
                protected_channels.get(name),
            )
            quantized_module = QuantizedLinear(
                module,
                scales,
                quantized_weight,
                protected_mask=protected_mask,
                fp_weight=fp_weight,
            )
            self._replace_module(model, name, quantized_module)

    def _quantize_linear_weight(self, weight, protected_indices=None):
        """
        Symmetric per-group quantization over the input dimension.
        Protected rows are kept in floating point during inference.
        """
        out_features, in_features = weight.shape
        max_q = 2 ** (self.w_bit - 1) - 1

        scales = torch.zeros_like(weight)
        quantized_weight = torch.zeros_like(weight, dtype=torch.int8)
        fp_weight = weight.detach().clone()

        protected_mask = torch.zeros(out_features, dtype=torch.bool)
        if protected_indices is not None and protected_indices.numel() > 0:
            protected_mask[protected_indices.long()] = True

        for out_idx in range(out_features):
            if protected_mask[out_idx]:
                continue

            for start in range(0, in_features, self.group_size):
                end = min(start + self.group_size, in_features)
                group = weight[out_idx, start:end]

                scale = group.abs().max() / max_q
                scale = scale.clamp_min(1e-8)

                q_group = torch.round(group / scale).clamp(-max_q, max_q).to(torch.int8)

                scales[out_idx, start:end] = scale
                quantized_weight[out_idx, start:end] = q_group

        return scales, quantized_weight, protected_mask, fp_weight

    def _replace_module(self, model, name, new_module):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)


class DemoMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def generate_demo_data(
    num_samples=256,
    input_dim=16,
    num_classes=3,
    batch_size=32,
):
    prototypes = F.normalize(torch.randn(num_classes, input_dim), dim=-1)
    labels = torch.randint(0, num_classes, (num_samples,))
    features = prototypes[labels] + 0.35 * torch.randn(num_samples, input_dim)

    batches = []
    for start in range(0, num_samples, batch_size):
        batches.append(features[start:start + batch_size])

    return features, labels, batches


def train_demo_model(model, features, labels, epochs=120, lr=3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0 or epoch == 1:
            accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} acc={accuracy:.3f}")


def evaluate_model(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = F.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, accuracy


def run_demo():
    torch.manual_seed(7)

    features, labels, calibration_data = generate_demo_data()

    print("=== Training Float Model ===")
    model = DemoMLP()
    train_demo_model(model, features, labels)

    float_loss, float_acc = evaluate_model(model, features, labels)
    print(f"float_loss={float_loss:.4f} float_acc={float_acc:.3f}")

    print("\n=== AWQ Quantization ===")
    quantizer = AWQQuantizer(w_bit=4, group_size=8, protect_ratio=0.125)
    quantized_model, protected_channels = quantizer.quantize_model(
        model,
        calibration_data=calibration_data,
    )

    quant_loss, quant_acc = evaluate_model(quantized_model, features, labels)
    print(f"quant_loss={quant_loss:.4f} quant_acc={quant_acc:.3f}")

    print("\n=== Protected Channels ===")
    for layer_name, channels in protected_channels.items():
        print(f"{layer_name}: {channels.tolist()}")


if __name__ == "__main__":
    run_demo()
