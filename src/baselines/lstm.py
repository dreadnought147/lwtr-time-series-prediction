import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path



class SequenceDataset(Dataset):
    """Turn a 1-D observation array into (window, target) supervised pairs."""

    def __init__(self, obs: np.ndarray, targets: np.ndarray, k: int):
        assert len(obs) == len(targets)
        self.k = k
        self.obs = torch.from_numpy(obs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.obs) - self.k

    def __getitem__(self, idx):
        window = self.obs[idx : idx + self.k].unsqueeze(-1)  # (k, 1)
        target = self.targets[idx + self.k]                  # scalar
        return window, target


#ourt model 
class LSTMForecaster(nn.Module):
    """Small LSTM: input sequence -> next-step return prediction."""

    def __init__(self, inputsize:int =1, hiddensze:int=32, numLayers:int=1):
        super().__init__()
        self.elestm = nn.LSTM(inputsize, hiddensze, numLayers, batch_first=True)
        self.head = nn.Linear(hiddensze,1) # what doe sthis do ?. 

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x: (batch, the sequence len, input_size)
        out, _ = self.elestm(x) #returns a tensor (batch, the sequence len, input_size)
        last = out[0:, -1, 0:] #what does this even return ? 
        # We take only the last timestep's hidden state (out[:, -1, :]) as the summary
        #  of the whole sequence, then a linear head projects it t0o a single scalar 
        # (the return forecast).
        return self.head(last).squeeze(-1) # batch,

def train_forecaster(model, dataset, epochs: int = 5, batchsze: int=64, learn_rate:float = 1e-3):
    """Standard PyTorch training loop with MSE loss."""

    load_the_data = DataLoader(dataset, batch_size=batchsze,shuffle=True ) #what does shuffling mean  ? 
    optimF = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = nn.MSELoss() #CREATES A MSELOSS CLASS

    # THE LOOP WE ARE NETERING 

    for epoch in range(epochs):
        cum_loss = 0.0
        for windows, targets in load_the_data:
            optimF.zero_grad()
            predictions = model(windows)
            loss = loss_fn(predictions, targets) # an array of our predictins vs the targets 
            loss.backward()
            optimF.step()
            cum_loss += loss.item() * len(targets) # why is it this ? 
        print(f"epoch {epoch+1}: avg mse = {cum_loss / len(dataset):.6f}")
    return model


def predictions_to_postsions(predictions: np.ndarray, thresh:float=0.0) -> np.ndarray:
    """Convert continuous forecasts to discrete positions {-1, 0, +1} via threshold."""
    positions =  np.zeros_like(predictions, dtype=np.int8)
    positions[predictions > thresh] = 1 #replace all value 0s with one at the smae index in predcitions in postions 
    #where teh value in predcitions is greatehr than teh threshold ? 
    # 
    positions[predictions < -thresh] = -1
    return positions


#  WE ARE HERE
def evaluate(positions: np.ndarray, returns: np.ndarray, transaction_cost: float =0.0) -> dict:
    """Compute total reward using the same reward formula as TimeSeriesEnv."""
    pnl = positions * returns
    flips = np.abs(np.diff(positions, prepend=0))
    cost = transaction_cost * flips
    reward = pnl - cost
    return {
        "total_reward": float(reward.sum()),
        "n_trades": int((flips>0).sum()),
        "n_positions" : int((positions != 0 ).sum())
    }

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.loader import load_aligned
    from src.data.transforms import to_env_arrays

    torch.manual_seed(0)

    obs, rets = to_env_arrays(load_aligned(), "vix_close", "sp_close")

    K = 5 #

    dataset = SequenceDataset(obs, rets, k=K) #more clarerity opn what K is 
    model = LSTMForecaster(inputsize=1, hiddensze=32)
    train_forecaster(model, dataset, epochs=3, batchsze=64)

    # Forecast the whole dataset (no shuffle so positions align with dates)
    model.eval()
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        preds = torch.cat([model(w) for w, _ in loader]).numpy()

    positions = predictions_to_postsions(preds, thresh=0.0)
    result = evaluate(positions, rets[K:], transaction_cost=0.0001)
    print(f"LSTM baseline: {result}")    


