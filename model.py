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
        print("In the len function")
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
        print(f"idx: {idx}, cumulative_windows: {cumulative_windows}, audio_file_idx: {audio_file_idx}")

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
train_files, test_files = train_test_split(audio_files, test_size=0.2, random_state=42)

# Create train and test datasets
train_dataset = AudioDataset(list(train_files), window_size, stride, data["stress_delta"])
test_dataset = AudioDataset(list(test_files), window_size, stride, data["stress_delta"])


# ### Training - basic without n-fold cross validation

# In[ ]:


# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model = model.to(device)
    model.train()
    train_loss = 0.0

    # DataLoader uses len function to get indices of sliding windows and then calls them (get_item)
    # shuffle=False, as data points depend on each other
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Iterate over training batches
    for batch, target in train_loader:
        batch = batch.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        # Forward pass
        #outputs = model(batch)
        hidden_states, stress_pred = model(batch.to(device))
        print("stress_pred size: ", stress_pred.size())
        print("target size: ", target.size())

        # Compute loss
        loss = criterion(stress_pred, target)
        train_loss += loss.item() * batch.size(0)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Compute average train loss for the epoch
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Evaluation on test set
    model.eval()
    test_loss = 0.0

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    with torch.no_grad():
        for batch, target in test_loader:
            batch = batch.to(device)
            target = target.to(device)

            # Forward pass
            hidden_states, stress_pred = model(batch)

            # Compute loss
            loss = criterion(stress_pred, target)
            test_loss += loss.item() * batch.size(0)

    # Compute average test loss for the epoch
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Plot loss over epochs
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/loss_plot.png')
plt.show()

# Save trained model
torch.save(model.state_dict(), '/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/trained_model.pt')
torch.cuda.empty_cache()


# In[ ]:


# Run this immediately if memory error above (only on vmc-mbp)
#gc.collect()
#torch.cuda.empty_cache()
# Is this really the solution?


# In[ ]:


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    """
    Predict stress or extract embeddings from raw audio signal.

    Args:
        x (np.ndarray): Input raw audio signal.
        sampling_rate (int): Sampling rate of the audio signal.
        embeddings (bool, optional): Flag to return stress level instead of emotions. Defaults to False.

    Returns:
        np.ndarray: Predicted emotions or embeddings.

    """
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]
    y = y.detach().cpu().numpy()
    return y


# #### Training advanced  - with n-fold cross validation

# In[ ]:


# Training loop
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model = None

# Create the cross-validation splits
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(skf.split(train_dataset.audio_files, train_dataset.targets)):
    # Create the train and test datasets for the current fold
    train_fold_dataset = torch.utils.data.Subset(train_dataset, train_index)
    test_fold_dataset = torch.utils.data.Subset(train_dataset, test_index)

    # Create the train and test data loaders
    train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_fold_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize the model for each fold
    model = StressModel.from_pretrained(model_name)
    model = model.to(device)

    # Initialize the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Iterate over training batches
        for batch, target in train_loader:
            batch = batch.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward pass
            #outputs = model(batch)
            outputs = model(batch.to(device))

            # Compute loss
            loss = criterion(outputs, target)
            train_loss += loss.item() * batch.size(0)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Compute average train loss for the epoch
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluation on test set
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch, target in test_loader:
                batch = batch.to(device)
                target = target.to(device)

                # Forward pass
                outputs = model(batch)

                # Compute loss
                loss = criterion(outputs, target)  # Adjust targets according to your data
                test_loss += loss.item() * batch.size(0)

        # Compute average test loss for the epoch
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

         # Save the best model based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model.state_dict()

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Plot loss over epochs
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Fold" + str(fold))
    plt.legend()
    plt.savefig('/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/loss_plot_fold_' + str(fold) + '.png')
    plt.show()

#TODO: Evaluate on the test data set (rename above test to validation data set)

# Save trained model
torch.save(best_model, '/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/trained_model_cv.pt')


# In[ ]:




