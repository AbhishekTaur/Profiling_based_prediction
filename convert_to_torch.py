from model import MLP
import torch

model = MLP(15, 16, 8)
model.load_state_dict(torch.load('checkpoint/MLP_model_019.pwf'))
model.eval()
data_X = [[7.78945338e+02, 1.10145554e+03, 1.78500882e+02, 2.26806641e-01, 2.48689490e-01, 9.80860579e+00,
           1.59348961e-01, 0, 0, 0, 1, 0, 0, 0, 0]]
X = torch.Tensor(data_X)

traced_script_module = torch.jit.trace(model, X)


traced_script_module.save("model.pt")

