import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        adapter_name: str = "default",
    ):
        # base Linear init: sets self.weight, self.bias, self.in_features, self.out_features
        super().__init__(in_features, out_features, bias=bias)

        # LoRA-related mappings (per adapter)
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}

        # Actual LoRA modules per adapter
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()

        # which adapter is used in forward()
        self.active_adapter = adapter_name

        # if r > 0, create the initial adapter
        if r > 0:
            self.update_layer(
                adapter_name=adapter_name,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
            )

    # -------------------------
    # Your update_layer (unchanged)
    # -------------------------
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(
            nn.ModuleDict({adapter_name: lora_dropout_layer})
        )

        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self.to(self.weight.device)

    # -------------------------
    # Extra small helpers
    # -------------------------
    def reset_lora_parameters(self, adapter_name: str):
        """Init LoRA weights: A ~ small random, B = 0 (standard LoRA init)."""
        r = self.r.get(adapter_name, 0)
        if r <= 0:
            return

        A = self.lora_A[adapter_name]
        B = self.lora_B[adapter_name]
        # A: Kaiming uniform, B: zeros
        nn.init.kaiming_uniform_(A.weight, a=5**0.5)
        nn.init.zeros_(B.weight)

    def set_adapter(self, adapter_name: str):
        """Switch which adapter is active during forward()."""
        self.active_adapter = adapter_name

    # -------------------------
    # Forward with LoRA
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base Linear forward
        result = F.linear(x, self.weight, self.bias)

        adapter_name = self.active_adapter

        # if this adapter exists and has rank > 0, apply LoRA update
        if (
            adapter_name in self.lora_A
            and adapter_name in self.lora_B
            and adapter_name in self.lora_dropout
            and self.r.get(adapter_name, 0) > 0
        ):
            lora_A = self.lora_A[adapter_name]
            lora_B = self.lora_B[adapter_name]
            lora_dropout = self.lora_dropout[adapter_name]
            scaling = self.scaling[adapter_name]

            # x -> dropout -> A -> B -> scaled additive update
            lora_update = lora_B(lora_A(lora_dropout(x))) * scaling
            result = result + lora_update

        return result


layer = LoRALinear(
    in_features=768,
    out_features=768,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    adapter_name="task_a",
)

x = torch.randn(4, 10, 768)
y = layer(x)  # uses adapter "task_a"

# Add another adapter later, reusing your update_layer as-is
layer.update_layer(
    adapter_name="task_b",
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    init_lora_weights=True,
)
layer.set_adapter("task_b")
y2 = layer(x)
