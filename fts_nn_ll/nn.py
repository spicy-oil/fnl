'''
Neural network - PeakIdentifier
Loss function -  FocalLoss 
training loop - train_model()
postprocessing - get_model_predictions()
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from .functions import preprocess_chunk, preprocess_interp, mad
from tqdm import tqdm
from scipy.signal import find_peaks


#%% Neural network
class PeakIdentifier(nn.Module):
    '''
    NN for line identification
    '''
    def __init__(self, hidden_size=64):
        super(PeakIdentifier, self).__init__()

        hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=2, bidirectional=True, batch_first=True)

        self.dense1 = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size * 2),  # +1 from wn
            nn.PReLU(),
        )

        self.dense2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.PReLU(),
        )

        self.output = nn.Linear(hidden_size * 2, 2)  # 2 classes at the moment

    def forward(self, x):
        # x shape (batch_size, 2, chunk_size)
        wn = x[:, 0, :]  # Shape (batch_size, chunk_size)
        x = x[:, 1, :]  # Shape (batch_size, chunk_size), x is now spec

        # Reshape input to match LSTM input format
        x = x.unsqueeze(2)  # Shape (batch_size, chunk_size, 1)
        # LSTM layers
        x, (hs, cs) = self.lstm(x) # Shape (batch_size, chunk_size, hidden_size * 2), x is now spec encoding

        # Concatenate LSTM output with wn
        wn = wn.unsqueeze(2)  # Shape (batch_size, chunk_size, 1)
        x = torch.concat([wn, x], dim=2) # Shape (batch_size, chunk_size, hidden_size * 2 + 1)

        # TimeDistributed-like FCNN decoder layers
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output(x)  # (batch_size, chunk_size, 2)

        return x

#%% Loss functions

class FocalLoss(nn.Module):
    '''
    Mean loss for all pnts in the whole batch
    '''
    def __init__(self, alpha=None, gamma=2.0, device='cpu'):
        '''
        alpha - tensor of true class weights to partially handle class imbalance 
        gamma - focusing parameter to down-weight easy examples, also tried gamma=3 & 5, results appeared very similar
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, inputs, targets, mask):
        '''
        inputs - not softmaxed probabilities (N_pnts_in_batch, num_classes)
        targets - integer class indices (N_pnts_in_batch)
        mask - tensor of 1s and 0s, used to ignore any loss contributions (here these are pnts near chunk edges)
        '''
        # Ensure inputs are not softmaxed before this point
        log_probs = F.log_softmax(inputs, dim=1) # (N_pnts_in_batch, 2)
        probs = torch.exp(log_probs) # (N_pnts_in_batch, 2)

        # Gather the log probabilities for the true class (targets are indices)
        targets = targets.view(-1, 1)  # Reshape to (N_pnts_in_batch, 1)
        log_probs_true = log_probs.gather(1, targets).view(-1)  # predicted log probability of the true class log(p_t), shape (N_pnts_in_batch,)
        probs_true = probs.gather(1, targets).view(-1)  # predicted probability of the true class p_t, shape (N_pnts_in_batch,)

        # Compute the focal loss factor (1 - p_t)^gamma
        focal_weight = (1 - probs_true) ** self.gamma

        # Apply class weights to handle class imbalance
        alpha_t = self.alpha[targets.view(-1)].to(self.device)
        focal_weight = focal_weight * alpha_t

        # Compute the final loss
        loss = -focal_weight * log_probs_true * mask
        return loss.mean()
    

#%% Custom dataset for loading

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]  # Number of samples

    def __getitem__(self, idx):
        # Return a sample and the corresponding target
        return self.X[idx, :], self.Y[idx, :]
    
#%% Training

def train_model(xy, line_region, model=None, chunk_size=1024, batch_size=32, num_epochs=64, lr=0.001, hidden_size=64, plot=False):
    '''
    Parameters
    ----------
    xy : training data
    line_region : needed to not train too many chunks that are purely noise
    model : PeakIdentifier instance, new one will be created if none specified
    batch_size : int, optional
        power of 2. The default is 32.
    num_epochs : int, optional
        The default is 64.

    Returns
    -------
    trained model with lowest test loss in eval() mode
    '''
    X1, X2, Y1 = preprocess_chunk(xy, chunk_size=chunk_size)  # (wn, spec, class, loss fn weights)
    
    # Train in line_region only, but also a few chunks with noise only (within +- 100 cm-1)
    idx = (X1[:, 0] > (line_region[0] - 100)) & (X1[:, 1] < (line_region[1] + 100))
    X1 = X1[idx] # wn
    X2 = X2[idx] # spec
    Y1 = Y1[idx] # class target

    # Normalise wn
    wn_min = X1.min()
    wn_max = X1.max()
    X1 = (X1 - wn_min) / (wn_max - wn_min)
    
    # Convert data to PyTorch tensors and move to GPU
    # Might want to change this if need to specify device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(np.array([X1, X2]), dtype=torch.float32).permute(1, 0, 2).to(device) # From shape ()
    Y_tensor = torch.tensor(Y1, dtype=torch.long).to(device)

    # Parameters
    batch_size = batch_size
    num_epochs = num_epochs

    # Split the data into training and testing sets (80% train, 20% test)
    train_size = int(0.8 * len(X_tensor))
    test_size = len(X_tensor) - train_size
    train_idx, test_idx = torch.utils.data.random_split(range(len(X_tensor)), [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # Split data into train and test
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    Y_train, Y_test = Y_tensor[train_idx], Y_tensor[test_idx]

    # Create DataLoader for training and testing datasets
    train_dataset = CustomDataset(X_train, Y_train)
    test_dataset = CustomDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if model is None:
        # Initialize the model
        model = PeakIdentifier(hidden_size).to(device)

    # Focal loss
    flattened_targets = Y_train.flatten()
    class_counts = np.bincount(flattened_targets.cpu())  # (N_class_0, N_class_1)
    total_samples = len(flattened_targets)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0, device=device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Set up counters
    lowest_test_loss = 1e10
    best_model = None
    best_model_precision = None
    best_model_recall = None
    best_model_epoch = None
    epochs_since_improvement = 0
    epochs = []
    
    train_losses = []
    train_accs = []
    train_precisions = []
    train_recalls = []
    train_F1s = []
    
    test_losses = []
    test_accs = []
    test_precisions = []
    test_recalls = []
    test_F1s = []
    
    # Training loops
    for epoch in range(num_epochs):
        # Training mode
        model.train()
        
        # Initialise counters
        train_loss = 0
        total_correct = 0
        total_samples = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        # Train over this epoch
        for inputs, targets in train_loader:
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Pnts within 16 pnts from the chunk edges do not contribute to loss, may need to increase if N_interp is high     
            mask = torch.ones_like(targets)
            mask[:, :16] = 0
            mask[:, -16:] = 0
            # Flatten to (N_pnts_in_batch)
            mask = mask.view(-1)
            
            # Loss calculations
            outputs = outputs.view(-1, 2) # (batch_size, chunk_size, 2) to (N_pnts_in_batch, 2)
            targets = targets.view(-1) # (batch_size, chunk_size) to (N_pnts_in_batch)
            loss = loss_fn(outputs, targets, mask)
            train_loss += loss.item()
            
            # Evaluation metrics related calculations using p_th = 0.5
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            true_positive += ((predicted == 1) & (targets == 1)).sum().item()
            false_positive += ((predicted == 1) & (targets == 0)).sum().item()
            false_negative += ((predicted == 0) & (targets == 1)).sum().item()

            loss.backward()
            optimizer.step()
            
        # Record metrics
        train_losses.append(train_loss / len(train_loader)) # record avg loss per pnt
        train_accuracy = total_correct / total_samples
        train_accs.append(train_accuracy)   
        train_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        train_precisions.append(train_precision)
        train_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        train_recalls.append(train_recall)
        train_F1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        train_F1s.append(train_F1)
        
        # Evaluation mode for testing with test dataset
        model.eval()
        
        # Initialise counters
        total_correct = 0
        total_samples = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        with torch.no_grad(): # not training
            test_loss = 0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                
                # Pnts within 16 pnts from the chunk edges do not contribute to loss, may need to increase if N_interp is high
                mask = torch.ones_like(targets)
                mask[:, :16] = 0
                mask[:, -16:] = 0
                # Flatten to (N_pnts_in_batch)
                mask = mask.view(-1)
                
                # Loss calculations
                outputs = outputs.view(-1, 2)
                targets = targets.view(-1)
                loss = loss_fn(outputs, targets, mask)
                test_loss += loss.item()
                
                # Evaluation metrics related calculations using p_th = 0.5
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                true_positive += ((predicted == 1) & (targets == 1)).sum().item()
                false_positive += ((predicted == 1) & (targets == 0)).sum().item()
                false_negative += ((predicted == 0) & (targets == 1)).sum().item()
        
        # Record metrics
        test_losses.append(test_loss / len(test_loader)) # record avg loss per pnt
        test_accuracy = total_correct / total_samples
        test_accs.append(test_accuracy)
        test_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        test_precisions.append(test_precision)
        test_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        test_recalls.append(test_recall)
        test_F1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
        test_F1s.append(test_F1)
        
        # Update best_model if test loss is lowest so far
        if test_loss / len(test_loader) < lowest_test_loss:
            lowest_test_loss = test_loss / len(test_loader)
            best_model = model
            best_model_precision = test_precision
            best_model_recall = test_recall
            best_model_epoch = epoch + 1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            
        # Count epochs for plotting
        epochs.append(epoch + 1)
        
        # Adaptive learning rate to fine tune towards performance plateau
        if epochs_since_improvement > 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= .9
        # Early stopping to save time
        if epochs_since_improvement > 20:
            break
        
        print(f'Epoch {epoch + 1}/{num_epochs}, ' + 
              f'train loss: {train_loss / len(train_loader):.4f}, ' +
              f'test loss: {test_loss / len(test_loader):.4f}, ' + 
              f'train prec: {train_precision:.4f}, ' +
              f'test prec: {test_precision:.4f}, ' +
              f'train recall: {train_recall:.4f}, ' +
              f'test recall: {test_recall:.4f}, ' +
              f'train F1: {train_F1:.4f}, ' +
              f'test F1: {test_F1:.4f}')
    
    print(f'Output model at epoch {best_model_epoch:.0f} with lowest test loss: {lowest_test_loss:.4f}, precision: {best_model_precision:.4f}, recall: {best_model_recall:.4f}', flush=True)
    
    # After all epochs
    if plot:
        # Loss-epoch plot
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 3), sharex=True, constrained_layout=True)
        ax1.plot(epochs, train_losses, color='k', ls=':', label='Train loss')
        ax1.plot(epochs, test_losses, color='k', ls='-', label='Test loss')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.set_xlabel('Epoch')
        ax2 = fig.add_axes([0.35, 0.5, 0.3, 0.4])
        ax2.plot(epochs, train_losses, color='k', ls=':', lw=.8, label='Train loss')
        ax2.plot(epochs, test_losses, color='k', ls='-', lw=.8, label='Test loss')
        best_epoch = epochs[np.argmin(abs(np.array(test_losses) - lowest_test_loss))]
        ax2.vlines(best_epoch, 0, 1, 'r')
        ax2.set_ylim(lowest_test_loss*0.8, lowest_test_loss*1.4)
        #fig.tight_layout()
        
        # Other metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        
        # Acc-epoch plot, train-test accs diverge after losses diverge
        ax1.plot(epochs, train_accs, color='k', ls=':', label='Train acc')
        ax1.plot(epochs, test_accs, color='k', ls='-', label='Test acc')
        ax1.set_ylabel('Acc')
        ax1.legend(loc='lower right')
        ax1.set_xlabel('Epoch')
        
        # Prec-epoch plot
        ax2.plot(epochs, train_precisions, color='k', ls=':', label='Train precision')
        ax2.plot(epochs, test_precisions, color='k', ls='-', label='Test precision')
        ax2.set_ylabel('Precision')
        ax2.legend(loc='lower right')
        ax2.set_xlabel('Epoch')
        
        # Recall-epoch plot
        ax3.plot(epochs, train_recalls, color='k', ls=':', label='Train recall')
        ax3.plot(epochs, test_recalls, color='k', ls='-', label='Test recall')
        ax3.set_ylabel('Recall')
        ax3.legend(loc='lower right')
        ax3.set_xlabel('Epoch')
        
        # F1-epoch plot
        ax4.plot(epochs, train_F1s, color='k', ls=':', label='Train F1-score')
        ax4.plot(epochs, test_F1s, color='k', ls='-', label='Test F1-score')
        ax4.set_ylabel('F1-score')
        ax4.legend(loc='lower right')
        ax4.set_xlabel('Epoch')
        
        fig.tight_layout()
        
        with torch.no_grad():
            N_test = 4 # 4 chunks to test and plot
            X_plot, Y_plot = test_dataset[:N_test] # first 4 shuffled test chunks for 4 plots
            output = best_model(X_plot)
            log_probs = F.log_softmax(output, dim=2)
            output = torch.exp(log_probs).cpu()
            for idx in range(N_test):
                wn = X_plot[idx, 0, :].cpu() * (wn_max - wn_min) + wn_min
                spec = X_plot[idx, 1, :].cpu()
                plt.figure()
                plt.plot(wn, spec, 'gray', lw=3, label='spec')
                plt.plot(wn, output[idx, :, 0], 'k',
                         lw=2, label='not peak prob.')
                plt.plot(wn, output[idx, :, 1], 'tab:green',
                         lw=2, label='peak prob.')
                plt.scatter(wn[Y_plot.cpu()[idx] == 1], spec[Y_plot.cpu()[
                            idx] == 1], color='tab:green', marker='^', s=100, label='Point closest to wn')
                plt.xlabel('wn (/cm)')
                plt.ylabel('snr')
                plt.legend(loc='upper right')

    return best_model

# %% Run prediction for real spectrum


def get_model_predictions(model, batch_size, wn, spec, npo, exp_res, line_region, chunk_size, N_interp, peak_prob_th, mad_scaling=False, plot=False):
    '''
    Interpolates spec, scale wn, and input into model
    All consecutive points with peak prob above peak_prob_th (p_th) are grouped and identified for lines
    Returns [peak positions, snr of peak positions, wn used for prediction, peak probability curve]
    Plotting interpolated spectrum and probabilities as whole could be very slow (e.g. 2 million pnts spectrum interpolated into 4 million)
    '''
    # Interpolate the same way simulation data was interpolated
    input_wn, input_spec = preprocess_interp(wn, spec, N=N_interp)

    # Scale input wn
    wn_min = input_wn.min()
    wn_max = input_wn.max()
    wn_norm_scale = 1
    input_wn_scaled = (input_wn - wn_min) * wn_norm_scale / (wn_max - wn_min)

    # Need to split spectrum into chunks with overlapping windows (50% are overlaps)
    # 50% of pnts in the first and final chunk of sequence of overlapping chunks do not have overlaps
    # This is fine because we don't have or are not interested in lines at the ends of each spectrum
    chunks = []
    chunk_prediction_size = int(chunk_size / 2)
    print('------------------------------------------------')
    print('Slicing up the spectrum and scaling noise for each slice')
    for i in tqdm(range(0, npo * int(2**N_interp), chunk_prediction_size)):
        if mad_scaling:
            # !!!!! HUMAN DECISIONs !!!!!
            # The MAD of Gaussian white noise is 0.675
            scaling = 0.6745 / mad(input_spec[i:i+chunk_size])
            if scaling > 1:  # only apply to raised noise levels
                scaling = 1
        else:
            scaling = 1
        chunks.append([input_wn_scaled[i:i+chunk_size], input_spec[i:i +
                      chunk_size] * scaling])
    # The final chunk is length chunk_prediction_size, does not fit a tensor, so remove it since it does not need to be evaluated as its at the end
    chunks = np.array(chunks[:-1])

    # Move to device to run through the model for predictions
    # Might want to change this if need to specify device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunks = torch.tensor(chunks, dtype=torch.float32).to(device)

    print('------------------------------------------------')
    print('Making predictions')
    chunks_predictions = []
    with torch.no_grad():
        for i in range(0, chunks.shape[0], batch_size):
            # Because model assumes batches of chunks, make chunks using batch_size
            batch = chunks[i:i+batch_size]
            # Move batch to GPU if using one
            # batch = batch.to(device)
            # Prediction
            predictions = model(batch)
            # Convert to probs
            log_probs = F.log_softmax(predictions, dim=2)
            predictions = torch.exp(log_probs)
            # Store predictions and move them back to CPU if using GPU
            chunks_predictions.append(predictions.cpu())

    # Concatenate all predictions into a single tensor
    predictions = torch.cat(chunks_predictions, dim=0) # Shape (total_chunks, chunk_size, 2) the final dim index as [not_peak_prob, peak_prob]

    chunks = chunks.cpu()
    # Scale wn back
    chunks[:, 0, :] = chunks[:, 0, :] * (wn_max - wn_min) + wn_min
    chunks = chunks.permute(0, 2, 1) # Shape (total_chunks, chunk_size, 2) the final dim index as [wn, spec]

    # Extract predictions using the overlapping window method, this might require chunk_size to be a multiple of 2
    chunks = chunks[:, int(chunk_prediction_size/2):-int(chunk_prediction_size/2), :]
    predictions = predictions[:, int(chunk_prediction_size/2):-int(chunk_prediction_size/2), :]

    # Don't need predictions outside line_region
    idx = (chunks[:, :, 0] > line_region[0]) & (chunks[:, :, 0] < line_region[1])
    chunks_relevant = chunks[idx]
    predictions_relevant = predictions[idx]

    if plot:
        # Slow plotting
        plt.figure()
        plt.hlines(0, line_region[0], line_region[1], 'k', alpha=0.3)
        plt.plot(chunks_relevant[:, 0], chunks_relevant[:, 1], 'gray', lw=2, label='stitched (and noise scaled if MAD=True) spec')
        plt.plot(chunks_relevant[:, 0], (predictions_relevant[:, 1] * 10), 'tab:green', lw=2, label=r'$p_1\times10$')
        plt.hlines(0.5 * 10, line_region[0], line_region[1], label='peak prob. = 0.5')
        plt.xlabel('wn (/cm)')
        plt.ylabel('snr')
        plt.legend(loc='upper right')

    # !!!!! HUMAN DECISION !!!!! p_th
    peak_prob_th = peak_prob_th
    
    # Extract groups of pnts above p_th
    above_threshold = np.where(predictions_relevant[:, 1] > peak_prob_th)[0]
    diff = np.diff(above_threshold)
    group_indices = np.split(above_threshold, np.where(diff > 1)[0] + 1)

    line_wn = []
    line_height = []
    
    for g in group_indices:  # group of probabilities consecutively above threshold, containing one or more line per group
        group_wn = chunks_relevant[g][:, 0].numpy()
        group_prob = predictions_relevant[g][:, 1].numpy()
        
        group_prob = np.pad(group_prob, 1) # pad with zeros to produce detectable peaks for find_peaks()
        peaks, _ = find_peaks(group_prob, height=(peak_prob_th, 1))
        group_wn = np.pad(group_wn, 1) # pad with zeros to produce indices for detectable peaks for find_peaks()
        group_wns = group_wn[peaks]

        height_idx = [np.argmin(abs(wn - gw)) for gw in group_wns]
        line_wn += group_wns.flatten().tolist()
        temp = spec[height_idx]
        temp[temp < 2] = 2 # initial guess for fitting is to be greater than 2 S/N
        line_height += temp.tolist() # the snr value of the classified wn point(s)

    # wn and snr used for initial guesses of Voigt profile fits to the spectrum for line list
    line_wn = np.array(line_wn)
    line_height = np.array(line_height)

    # Combine predictions and plot
    if plot:
        spec_idx = (wn > line_region[0]) & (wn < line_region[1])
        plt.figure()
        plt.title('Detected lines')
        plt.hlines(0, line_region[0], line_region[1], 'k', alpha=0.3)
        plt.plot(wn[spec_idx], spec[spec_idx], 'k', lw=2, label='spec')
        plt.vlines(line_wn, 0, line_height, 'tab:green',
                   lw=2, label='detected lines')
        plt.xlabel('wn (/cm)')
        plt.ylabel('snr')
        plt.legend(loc='upper right')

    return [line_wn, # wn of detected lines
            line_height, # snr of detected lines
            chunks_relevant[:, 0].cpu().numpy(), # wn used for prediction 
            predictions_relevant[:, 1].cpu().numpy()] # peak probability curve
