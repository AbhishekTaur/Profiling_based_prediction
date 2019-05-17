from model import MLP
import torch

model = MLP(15, 16, 8)
model.load_state_dict(torch.load('checkpoint/MLP_model_019_train.pwf'))
model.eval()
data_X = [[[2.78631504e-06, 6.38972676e-06, 3.26745108e-06, 7.03612108e-03, 8.65014990e-01, 3.56484887e-05, 2.34573321e-01, 0, 0, 0, 1, 0, 0, 0, 0]]]
X = torch.Tensor(data_X)

traced_script_module = torch.jit.trace(model, X)


traced_script_module.save("model.pt")

