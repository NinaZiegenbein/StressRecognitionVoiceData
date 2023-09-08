#!/usr/bin/env python
# coding: utf-8

# This file implements Cross Validation or Leave-one-participant-out validation and uses MSE loss. Parser Arguments are also implemented.

# In[ ]:


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
from sklearn.model_selection import KFold, LeaveOneOut
import math
import glob
import re
from pydub import AudioSegment
import librosa
import os
import argparse
import datetime


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


# ## Load Pre-trained model

# In[ ]:


# load model from hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)


# In[ ]:


model = StressModel.from_pretrained(model_name)
model = nn.DataParallel(model)


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
    cov_pred_target = torch.mean((predictions - torch.mean(predictions)) * (targets - torch.mean(targets)))
    ccc = 2 * cov_pred_target / (torch.var(predictions) + torch.var(targets) + (torch.mean(predictions) - torch.mean(targets))**2 + torch.finfo(torch.float32).eps) # add epsilon to avoid divisions by zero
    # subtract from 1, as ccc is 1 for perfect agreement, and we want to minimize
    return 1. - ccc


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
window_size = 8  # seconds
stride = int(window_size / 2)
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
n_folds = 5
muse = False # True for using Muse Dataset, false for tsst v dst dataset
panel = True
use_valence = True


# In[ ]:


print("Trying parser arguments")

parser = argparse.ArgumentParser(description="Training stress data")

parser.add_argument('--data', type=int, default=2, help='2, if use MUSE dataset, 1, if use TSST dataset with panel ground truth, 0 if use TSST dataset with self-assessed stress levels')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate, Default 1e-4')
parser.add_argument('--window_size', type=int, default=6, help="Window size in seconds, default 6 seconds")
args = parser.parse_args()


if args.data == 0:
    muse = False
    panel = False
elif args.data == 1:
    muse = False
    panel = True
elif args.data == 2:
    muse = True
    panel = False
print("muse:", muse, "panel:", panel)
learning_rate = args.lr
window_size = args.window_size
stride = int(window_size / 2)


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
        frame_rate = int(audio.getframerate())

        # get number of windows in previous files
        windows_in_prev_files = sum(num_windows(AudioFileClip(audio_file).duration) for audio_file in self.audio_files[0:audio_file_idx])
        # get number of windows before idx in current file
        windows_in_file = (idx - windows_in_prev_files)
        start = int(windows_in_file * stride) # multiply with stride to get starting point in seconds
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


def convert_to_pcm(input_file, output_file):
    audio = AudioSegment.from_wav(input_file)
    audio.export(output_file, format="wav", parameters=["-ac", "1", "-ar", "44100"])


# In[ ]:


# Datset for Muse Challenge
class MuseDataset(Dataset):
    def __init__(self, muse_dict, window_size, stride):
        self.audio_files = list(muse_dict.keys())
        self.window_size = window_size
        self.stride = stride
        self.targets = muse_dict

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
        file_name = os.path.basename(audio_file)
        output_file = "/data/tsst_22_muse_video_nt_lab/processed/wav_converted/" + file_name
        convert_to_pcm(audio_file, output_file)

        """
        # load using librosa and resample to 16kHz
        a, sr = librosa.load(audio_file, sr=16000)
        frame_rate = sr

        # get number of windows in previous files
        windows_in_prev_files = sum(num_windows(librosa.get_duration(path=audio_file)) for audio_file in self.audio_files[0:audio_file_idx])
        # get number of windows before idx in current file
        windows_in_file = (idx - windows_in_prev_files)
        start = windows_in_file * stride  # multiply with stride to get starting point in seconds

        # Calculate the frame indices for the desired window
        start_frame = int(start * frame_rate)
        end_frame = start_frame + int(self.window_size * frame_rate)

        # Ensure the end frame is within audio bounds
        if end_frame <= len(a):
            window = a[start_frame:end_frame]
        else:
            # Pad the audio array with zeros
            print("padding!")
            num_frames_left = len(a) - start_frame
            zero_padding = np.zeros(int(self.window_size * frame_rate) - num_frames_left)
            window = np.concatenate((a[start_frame:], zero_padding))


        """
        audio = wave.open(output_file)
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

        # calculate the target values:
        valence, arousal = self.targets[audio_file]
        start_ms = start * 1000 # get start in milliseconds
        end_ms = start_ms + self.window_size * 1000

        if use_valence:
            stress_values = valence
        else:
            stress_values = arousal

        # Round start and end values to closest timestamp
        rounded_start = stress_values['timestamp'].iloc[(stress_values['timestamp'] - start_ms).abs().idxmin()]
        rounded_end = stress_values['timestamp'].iloc[(stress_values['timestamp'] - end_ms).abs().idxmin()]

        # Filter DataFrame to include rows between rounded start and end values
        filtered_df = stress_values[(stress_values['timestamp'] >= rounded_start) & (stress_values['timestamp'] <= rounded_end)]

        # Calculate the average of the values within the filtered DataFrame
        target = filtered_df['value'].mean()
        #print("window:", window.size(), "target:", target)
        #print("audio_file", audio_file_idx, "start_frame:", start_frame, "end_frame:", end_frame, )
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


# Muse22 data
if muse:
    print("The complete muse dataset is to big to use cross-validation. Either reduce amount of data or prepare for long runtimes (~30h per fold)")

# tsst data with panel assessed stress
elif panel:
    # Load your list of audio file paths
    data = pd.read_csv("/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/tsst_data.csv")
    audio_files = data["TSST_audio_segment"]

    # normalize labels between 0 and 1
    data['panel_stress_speech_average'] = data['panel_stress_speech_average'] / 100.0

    # Perform train-test split
    train_files, test_files, train_targets, test_targets = train_test_split(audio_files, data["panel_stress_speech_average"], test_size=0.1, random_state=19)
    print("test files", list(test_files))

    train_targets = train_targets.reset_index(drop=True)
    test_targets = test_targets.reset_index(drop=True)

    # Create test datasets
    test_dataset = AudioDataset(list(test_files), window_size, stride, test_targets)

# tsst data with self-assessed stress
else:
    # Load your list of audio file paths
    data = pd.read_csv("/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/tsst_data.csv")
    audio_files = data["TSST_audio_segment"]

    # normalize labels between -1 and 1
    data['stress_delta_scaled'] = data['stress_delta'] / 100.0

    # Perform train-test split -> smaller test size, since cross validation
    train_files, test_files, train_targets, test_targets = train_test_split(audio_files, data["stress_delta_scaled"], test_size=0.1, random_state=19)
    print("test files", test_files)

    train_targets = train_targets.reset_index(drop=True)
    test_targets = test_targets.reset_index(drop=True)

    # Create test datasets
    test_dataset = AudioDataset(list(test_files), window_size, stride, test_targets)


# ### Training - with n-fold cross validation

# In[ ]:


print("Training with", len(train_files) + len(test_files), "files")
print("window_size:", window_size, "; stride:", stride)
print("batch_size:", batch_size, "; learning_rate:", learning_rate)
#print(n_folds, "-fold cross validation")
if muse:
    print("Using MUSE data")
elif panel:
    print("Using TSST v DST Data with panel-assessed stress")
else:
    print("Using TSST v DST Data with self-assessed stress")
print("Using" + str(n_folds) +"-fold Cross-vaildation")

# Training loop
best_test_loss = float('inf')
best_model = None

# Create the cross-validation/leave-one-out splits
# kf = KFold(n_splits=n_folds, shuffle=True, random_state=19)
kf = LeaveOneOut()
for fold, (train_index, valid_index) in enumerate(kf.split(train_files, train_targets)):
    print(fold, valid_index)
    print(list(torch.utils.data.Subset(list(train_files), valid_index)))
test_losses_complete = []
# Iterate over the folds
for fold, (train_index, valid_index) in enumerate(kf.split(train_files, train_targets)):
    train_losses = []
    valid_losses = []
    baseline_losses = []
    test_losses = []
    # Create the train and test datasets for the current fold
    train_fold_files = torch.utils.data.Subset(list(train_files), train_index)
    train_fold_targets = torch.utils.data.Subset(list(train_targets), train_index)
    valid_fold_files = torch.utils.data.Subset(list(train_files), valid_index)
    valid_fold_targets = torch.utils.data.Subset(list(train_targets), valid_index)

    #train_fold_targets = train_fold_targets.reset_index(drop=True)
    #valid_fold_targets = valid_fold_targets.reset_index(drop=True)

    train_dataset = AudioDataset(list(train_fold_files), window_size, stride, train_fold_targets)
    valid_dataset = AudioDataset(list(valid_fold_files), window_size, stride, valid_fold_targets)

    # Initialize the model for each fold
    model = StressModel.from_pretrained(model_name)
    model = nn.DataParallel(model)
    model = model.to(device)

    # Initialize the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Iterate over epochs
    for epoch in range(num_epochs):
        print("start epoch", epoch)
        model.train()
        train_loss = 0.0
        mean_target = 0.0
        mean_stress_pred = 0.0

        # Create the train data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

        # Iterate over training batches
        for batch, target in train_loader:
            batch = batch.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward pass
            hidden_states, stress_pred = model(batch.to(device))

            # Compute loss
            loss = criterion(stress_pred, target)
            print("loss", loss, loss.item())
            print("pred:", stress_pred)
            print("target:", target)
            train_loss += loss.item() * batch.size(0)
            mean_target += torch.sum(target).item()
            mean_stress_pred += torch.sum(stress_pred).item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Compute average train loss for the epoch
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        mean_stress_pred /= len(train_loader.dataset)
        print("mean stress prediction:", mean_stress_pred)

        # Evaluation on validation set
        model.eval()
        valid_loss = 0.0
        baseline_loss = 0.0

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

        print("validation")
        with torch.no_grad():
            for batch, target in valid_loader:
                batch = batch.to(device)
                target = target.to(device)

                # Forward pass
                hidden_states, stress_pred = model(batch)
                print("pred:", stress_pred)
                print("target:", target)

                # Compute loss
                loss = criterion(stress_pred, target)  # Adjust targets according to your data
                valid_loss += loss.item() * batch.size(0)

                # compute baseline loss
                mean_pred = torch.full_like(target, mean_target) # make mean_target into size of target batch
                loss = criterion(mean_pred, target)
                baseline_loss += loss.item() * batch.size(0)

        # Compute average validation loss for the epoch
        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)

        baseline_loss /= len(valid_loader.dataset)
        baseline_losses.append(baseline_loss)

        # Evaluation on test set
        model.eval()
        test_loss = 0.0
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

        # Compute average test loss for the entire test dataset
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # Print progress
        print(f"Fold {fold+1}/{n_folds}, Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Test Loss: {test_loss: .4f}, Baseline Loss: {baseline_loss: .4f}")

    test_losses_complete.append(test_losses)
    # Save trained model
    torch.save(model.state_dict(), '/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/best_trained_model_cv_fold' + str(fold)+ "_" + str(datetime.date.today()) + '.pt')

    # Plot loss over epochs
    try:
        plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs+1), valid_losses, label='Validation Loss')
        plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
        plt.plot(range(1, num_epochs+1), baseline_losses, label='Baseline Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Fold" + str(fold+1))
        plt.legend()
        plt.savefig('/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/loss_plot_fold_' + str(fold) + '_' + "panel=" + str(panel) + "_" + str(datetime.date.today()) + '.png')
        plt.show()
    except:
        print("Error in plot")
        print("train_losses:", train_losses)
        print("validation losses:", valid_losses)
        print("test losses", test_losses)
        print("baseline losses", baseline_losses)

try:
    for fold in range(n_folds):
        plt.plot(range(1, num_epochs+1), test_losses_complete[fold], label=str('Test Loss Fold'+str(fold+1)))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Fold" + str(fold+1))
    plt.legend()
    plt.savefig('/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/loss_plot_fold_overview_' + "panel=" + str(panel) + "_" + str(datetime.date.today()) + '.png')
    plt.show()
except:
    print("Error in creating plot!")
    print("complete test losses: ", test_losses_complete)



# # Look at predictions of pretrained models

# In[ ]:


model_pt = torch.load('/data/dst_tsst_22_bi_multi_nt_lab/processed/audio_files/best_trained_model_cv_fold4_2023-09-06.pt')
train_dataset = AudioDataset(list(train_files), window_size, stride, train_targets)
input_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
for batch, target in input_loader:
    predictions = model(batch)
    print(predictions, target)


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()


# # Dumpster

# In[ ]:


# Now unnecessary downsampling function (dumpster)
def downsample_wav(input_path, output_path, target_sample_rate=16000):
    with wave.open(input_path, 'r') as input_wav:
        # Get input WAV properties
        num_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        original_sample_rate = input_wav.getframerate()
        total_frames = input_wav.getnframes()

        # Open the output WAV file
        with wave.open(output_path, 'w') as output_wav:
            output_wav.setnchannels(num_channels)
            output_wav.setsampwidth(sample_width)
            output_wav.setframerate(target_sample_rate)

            # Read and resample the input frames
            input_frames = input_wav.readframes(total_frames)
            resampled_frames, _ = audioop.ratecv(input_frames, sample_width, num_channels, original_sample_rate, target_sample_rate, None)

            # Write the resampled frames to the output file
            output_wav.writeframes(resampled_frames)

