#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import audeer
import audonnx
import audinterface
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from moviepy.editor import AudioFileClip
import wave
from sklearn.model_selection import StratifiedKFold
import math
import gc


# # Stress Model

# In[ ]:


class RegressionHead(nn.Module):
    """
    This class defines the regression head for the stress model. It consists of a series of linear layers
    and applies a linear transformation to the input features without any activation function.
    """
    def __init__(self, config):
        """
        @param config: model configuration
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.fc = nn.Linear(config.hidden_size, 1) # we only want a single stress value

    def forward(self, features, **kwargs):
        """
        Forward pass of the regression head.
        @param features: the hidden states
        @return: Output tensor after passing through the regression head
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        # x = torch.tanh(x) # Don't use actiavtion
        x = self.dropout(x)
        x = self.fc(x)

        return x


# In[ ]:


class StressModel(Wav2Vec2PreTrainedModel):
    """
    This class defines the speech stress classifier model based on the Wav2Vec2 architecture. It consists of
    the Wav2Vec2 base model, a regression head, and model weights.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        """
        Forward pass of the stress model.
        @param input_values: Raw input audio
        @return: Hidden states and logits
        """
        input_values = input_values.float()
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        stress_pred = self.classifier(hidden_states)
        return hidden_states, stress_pred


# ### Load Pre-trained model

# In[ ]:


# load model from hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = StressModel.from_pretrained(model_name)


# In[ ]:


# Freeze CNN layers, but not TransformerLayers
for name, param in model.named_parameters():
    if "wav2vec2.feature_extractor" in name:
        param.requires_grad = False
    if "wav2vec2.masked_spec_embed" in name:
        param.requires_grad = False


# In[ ]:


# additional loss infos/functions
def calculate_mae(predictions, targets):
    loss_func = nn.L1Loss()
    return loss_func(predictions, targets).item() * batch.size(0)

def calculate_rmse(predictions, targets):
    loss_func = nn.MSELoss()
    mse = loss_func(predictions, targets)
    return torch.sqrt(mse).item() * batch_size

def calculate_ccc(predictions, targets):
    cov_pred_target = torch.mean((predictions - predictions.mean()) * (targets - targets.mean()))
    ccc = 2 * cov_pred_target / (torch.var(predictions) + torch.var(targets) + (predictions.mean() - targets.mean())**2)
    return ccc.item() * batch.size(0)


# In[ ]:


# Create a new optimizer for the trainable layers (only transformer layers)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4 # learning rate according to their documentation
)

# loss function
criterion = nn.MSELoss()


# ### Load Data

# In[ ]:


# Set hyperparameters
window_size = 6  # seconds
stride = 3  # seconds
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
n_folds = 3


# In[ ]:


def num_windows(duration):
    """
    This helper function return the number of windows of a single file including padding for datapoints that do not perfectly fit into the window at the end
    @param duration: duration of the audio file
    @return: number of windows in that file
    """
    return math.ceil((duration-window_size)/stride)+1


# In[ ]:


# Define your custom dataset
class AudioDataset(Dataset):
    def __init__(self, audio_files, window_size, stride, targets):
        self.audio_files = audio_files
        self.window_size = window_size
        self.stride = stride
        self.targets = targets

    def __len__(self):
        # get duration of ALL FILES divided without rest by stride and summed together
        # returns the number of windows
        return sum(num_windows(AudioFileClip(audio_file).duration)
                   for audio_file in self.audio_files)

    def __getitem__(self, idx):
        # first we need to find the corresponding audio file
        cumulative_windows = 0
        audio_file_idx = 0

        while cumulative_windows <= idx:
            cumulative_windows += num_windows(AudioFileClip(self.audio_files[audio_file_idx]).duration)
            audio_file_idx += 1

        audio_file_idx -= 1  # since while is false, subtract the last plus

        audio_file = self.audio_files[audio_file_idx]
        audio = wave.open(audio_file)
        frame_rate = audio.getframerate()

        # get number of windows in previous files
        windows_in_prev_files = sum(num_windows(AudioFileClip(audio_file).duration) for audio_file in self.audio_files[0:audio_file_idx])
        # get number of windows before idx in current file
        windows_in_file = (idx - windows_in_prev_files)
        start = windows_in_file * stride # multiply with stride to get starting point in seconds
        audio.setpos(start)
        # check if window is bigger than amount of audio file "left"
        if start + int(self.window_size * frame_rate) <= audio.getnframes():
            window = audio.readframes(int(self.window_size * frame_rate))
        else:
            # pad the audio file with zeros
            num_frames_left = audio.getnframes() - start
            zero_padding = b'\x00' * (int(self.window_size * frame_rate) - num_frames_left)
            window = audio.readframes(num_frames_left) + zero_padding
        audio.close()

        # Convert the raw audio frames to a numeric representation
        samples = np.frombuffer(window, dtype=np.int16)

        # Convert samples to a tensor
        window = torch.tensor(samples)

        target = self.targets[audio_file_idx]
        return window, target


# In[ ]:


def custom_collate_fn(batch):
    # Separate the windows and targets from the batch
    windows, targets = zip(*batch)

    # Convert windows to tensors and concatenate them along the batch dimension
    windows = torch.stack(windows)

    # Convert targets to tensors
    targets = torch.tensor(targets, dtype=torch.float32)

    return windows, targets


# In[ ]:


# Load your list of audio file paths
data = pd.read_csv("/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/tsst_data.csv")
audio_files = data["TSST_audio_segment"]

# Perform train-test split
train_files, test_files, train_targets, test_targets = train_test_split(audio_files, data["stress_delta"], test_size=0.2, random_state=42)

train_targets = train_targets.reset_index(drop=True)
test_targets = test_targets.reset_index(drop=True)

# Create train and test datasets
train_dataset = AudioDataset(list(train_files), window_size, stride, train_targets)
test_dataset = AudioDataset(list(test_files), window_size, stride, test_targets)


# ### Training - basic without n-fold cross validation

# In[ ]:


print("Training with", len(audio_files), "files")
print("window_size:", window_size, "; stride:", stride)
print("batch_size:", batch_size, "; learning_rate:", learning_rate)
#print(n_folds, "-fold cross validation")

print("Using MSE Loss, shuffle=true, normal learning rate")

# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model = model.to(device)
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    train_rmse = 0.0

    # DataLoader uses len function to get indices of sliding windows and then calls them (get_item)
    # shuffle=False, as data points depend on each other
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Iterate over training batches
    for batch, target in train_loader:
        batch = batch.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        # Forward pass
        #outputs = model(batch)
        hidden_states, stress_pred = model(batch.to(device))

        # Compute loss
        loss = criterion(stress_pred, target)
        train_loss += loss.item() * batch.size(0)

        train_mae += calculate_mae(stress_pred, target)
        train_rmse += calculate_rmse(stress_pred, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Compute average train loss for the epoch
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    train_mae /= len(train_loader.dataset)
    train_rmse /= len(train_loader.dataset)

    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_rmse = 0.0

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    with torch.no_grad():
        for batch, target in test_loader:
            batch = batch.to(device)
            target = target.to(device)

            # Forward pass
            hidden_states, stress_pred = model(batch)

            # Compute loss
            loss = criterion(stress_pred, target)
            test_loss += loss.item() * batch.size(0)

            test_mae += calculate_mae(stress_pred, target)
            test_rmse += calculate_rmse(stress_pred, target)

    # Compute average test loss for the epoch
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    test_mae /= len(train_loader.dataset)
    test_rmse /= len(train_loader.dataset)

    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    print(f"Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

# Plot loss over epochs
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/loss_plot_mse.png')
plt.show()

# Save trained model
torch.save(model.state_dict(), '/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/model_mse.pt')
torch.cuda.empty_cache()

