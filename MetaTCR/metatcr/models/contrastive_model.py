import torch

class ContrastiveModel(torch.nn.Module):
    def __init__(self, input_dim = 96, output_dim = 96):
        super(ContrastiveModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, output_dim)

    def forward(self, x1, x2):
        x1, x2 = x1.float(), x2.float()
        out1 = torch.relu(self.fc1(x1))
        out2 = torch.relu(self.fc1(x2))

        out1_mean = torch.mean(out1, dim=1)
        out2_mean = torch.mean(out2, dim=1)
        out1_final = self.fc2(out1_mean)
        out2_final = self.fc2(out2_mean)

        return out1_final, out2_final

## get embedding from CL model. input data is a set of instances (bag)
# def get_bag_ebd(data_tensor, model):  ## data_tensor: tensor with size=(N, ebd_dim), model: ContrastiveModel
#
#     with torch.no_grad():
#         embeddings = []
#
#         for d in data_tensor:
#             out = model(d.unsqueeze(0), d.unsqueeze(0))
#             out1_final = out[-1]
#             embeddings.append(out1_final.squeeze(0))
#
#         ## list to tensor
#         embeddings = torch.stack(embeddings, dim=0)
#
#         return embeddings