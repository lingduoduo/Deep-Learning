import torch

class MyModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    @torch.jit.script_method
    def forward(self, input):
        if bool(input.sum() > 0):
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output

my_script_module = MyModule(2, 3)
