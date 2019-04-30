from model import MLP
import torch

model = MLP(7, 16, 8)
data_X = [[7.78945338e+02,1.10145554e+03, 1.78500882e+02,2.26806641e-01,2.48689490e-01,9.80860579e+00,1.59348961e-01]]
X = torch.Tensor(data_X)

traced_script_module = torch.jit.trace(model, X[0])


traced_script_module.save("model.pt")

