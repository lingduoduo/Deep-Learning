import torch
import torchvision
import torch.nn as nn

# 1. Prepare your model (e.g., pretrained ResNet-18) and set to evaluation mode
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 2. If using GPU/CPU, move model appropriately
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. Create an example (dummy) input that matches the expected shape & dtype
#    Use the same device as model
example_input = torch.randn(1, 3, 224, 224, device=device)

# 4. Trace the model to get a ScriptModule
#    It’s very important that the model is in eval() mode (so that layers like BatchNorm/Dropout behave in inference mode)
traced_model = torch.jit.trace(model, example_input)
#    Optionally you can check that the traced model behaves similarly on some inputs:
with torch.no_grad():
    original_out = model(example_input)
    traced_out   = traced_model(example_input)
    # You could assert approx equal, e.g., torch.allclose(original_out, traced_out, atol=1e-5)

# 5. Save the traced ScriptModule to a file
traced_model_file = "resnet18_traced.pt"
traced_model.save(traced_model_file)
print(f"Saved traced model to {traced_model_file}")

# 6. Later (or in production), load the saved ScriptModule for inference
loaded = torch.jit.load(traced_model_file, map_location=device)
loaded.eval()  # might still be good practice (though ScriptModule saved with eval-mode behavior)

# 7. Run inference on new data
#    Suppose `input_tensor` is your pre-processed input (batch size = 1)
input_tensor = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    output = loaded(input_tensor)
    # e.g., apply softmax or argmax for classification
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    print("Predicted class:", predicted_class.item())

