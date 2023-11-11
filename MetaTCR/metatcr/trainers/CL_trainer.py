import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


# Define training and evaluation functions
class CL_Trainer:

    @staticmethod
    def compute_loss(out1, out2, out3, loss_type="triplet", margin=0.1, alpha=1e-4):
        if loss_type == "triplet":
            cos_sim_pos = torch.nn.functional.cosine_similarity(out1, out2) - alpha
            cos_sim_neg = torch.nn.functional.cosine_similarity(out1, out3) + alpha
            loss = torch.relu(margin - cos_sim_pos + cos_sim_neg).mean()
        elif loss_type == "cosine":
            cos_sim_pos = torch.nn.functional.cosine_similarity(out1, out2)
            cos_sim_neg = torch.nn.functional.cosine_similarity(out1, out3)
            loss = -cos_sim_pos + cos_sim_neg + 2 * alpha
            loss = loss.mean()
        else:
            raise ValueError("Invalid loss_type. Choose between 'triplet' and 'cosine'.")

        return loss

    @staticmethod
    def train(model, dataloader, optimizer, device = "cuda" if torch.cuda.is_available() else "cpu", loss_type="triplet", margin=0.1):
        model.train()
        model.to(device)
        total_loss = 0.0
        scaler = GradScaler()  # for automatic mixed precision training
        progress_bar = tqdm(dataloader, desc="Training")

        for pos_sample1, pos_sample2, neg_sample in progress_bar:
            pos_sample1, pos_sample2, neg_sample = pos_sample1.to(device), pos_sample2.to(device), neg_sample.to(device)
            optimizer.zero_grad()

            with autocast():  # for automatic mixed precision training
                out1, out2 = model(pos_sample1, pos_sample2)
                _, out3 = model(pos_sample1, neg_sample)
                loss = CL_Trainer.compute_loss(out1, out2, out3, loss_type=loss_type, margin=margin)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        return total_loss / len(dataloader)

    @staticmethod
    def evaluate(model, dataloader, device = "cuda" if torch.cuda.is_available() else "cpu"):
        model.eval()
        model.to(device)
        with torch.no_grad():
            correct = 0
            total = 0
            progress_bar = tqdm(dataloader, desc="Evaluating")

            for pos_sample1, pos_sample2, neg_sample in progress_bar:
                pos_sample1, pos_sample2, neg_sample = pos_sample1.to(device), pos_sample2.to(device), neg_sample.to(
                    device)
                out1, out2 = model(pos_sample1, pos_sample2)
                _, out3 = model(pos_sample1, neg_sample)

                cos_sim_pos = torch.nn.functional.cosine_similarity(out1, out2)
                cos_sim_neg = torch.nn.functional.cosine_similarity(out1, out3)

                correct += torch.sum(cos_sim_pos > cos_sim_neg).item()
                total += pos_sample1.size(0)
                progress_bar.set_postfix({"accuracy": correct / total})

        return correct / total