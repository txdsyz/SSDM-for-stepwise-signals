#%%
'''

Acknowledgement:
The core architecture of this project and its three key supporting files (“module.py”, “util.py”, “diffusion.py”)
are derived from or modified based on the following open-source projects:
Choi, S. yet-another-pytorch-tutorial-v2, GitHub: https://github.com/sjchoi86/yet-another-pytorch-tutorial-v2

This file primarily utilizes the DiffusionUNetLegacy model defined by the modules mentioned above and associated diffusion constants to perform denoising evaluation and transition point detection for stepwise signals.


1.Please confirm that "module.py", "util.py", "diffusion.py", and this file are located in the same directory.
2.Please modify the paths in this script according to your local directory structure.


'''

#%%
import os
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

# Ensure util.py and diffusion.py are in the same directory
from util import (
    print_model_parameters,
    gp_sampler,
    get_torch_size_string,
    plot_1xN_torch_traj_tensor,
    periodic_step,
    plot_ddpm_1d_result,
)
from diffusion import (
    get_ddpm_constants,
    plot_ddpm_constants,
    DiffusionUNetLegacy,
    forward_sample,
    eval_ddpm_1d,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dc = get_ddpm_constants(
    schedule_name='cosine',
    T=1000,
    np_type=np.float32,
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#%%

class StepwiseDataset(Dataset):
    def __init__(self, root_dir, selected_ids=None):
        self.samples = []


        if selected_ids is None:
            selected_ids = list(range(1, 101))  #Each folder in this dataset contains 100 samples.

        subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        print(f" {len(subfolders)} files")

        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            for file_num in selected_ids:
                clean_name = f'clean_sample_{file_num}.csv' #noise-free sample
                noisy_name = f'noisy_clean_sample_{file_num}.csv' #noisy sample

                clean_path = os.path.join(subfolder_path, clean_name)
                noisy_path = os.path.join(subfolder_path, noisy_name)

                if os.path.exists(clean_path) and os.path.exists(noisy_path):
                    self.samples.append((clean_path, noisy_path))


        if len(self.samples) == 0:
            print(f"No valid data pairs found in {root_dir}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.samples[idx]
        #conver data into tensor
        clean_df = pd.read_csv(clean_path, header=None)
        noisy_df = pd.read_csv(noisy_path, header=None)

        clean_trace = torch.tensor(clean_df.iloc[:, 0].values, dtype=torch.float32).unsqueeze(0)
        noisy_trace = torch.tensor(noisy_df.iloc[:, 0].values, dtype=torch.float32).unsqueeze(0)

        return noisy_trace, clean_trace, noisy_path

#%% md
# Enhanced Loss Funciton
#%%
def compute_enhanced_loss(noise_pred, noise, x_0_batch, amp_factor, edge_factor):

    # Base loss function
    mse_loss = F.smooth_l1_loss(noise_pred, noise, reduction='none')  #Smooth L1 Loss

    # Amplitude weights
    amp_weights = 1.0 + amp_factor * torch.abs(noise)

    # Difference for Edge weights
    grad_x = torch.abs(torch.diff(x_0_batch, dim=2, append=x_0_batch[:, :, -1:]))  #First-order difference
    grad2_x = torch.abs(torch.diff(grad_x, dim=2, append=grad_x[:, :, -1:]))  #Second-order difference

    # Use a sliding window of size 3
    conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False).to(x_0_batch.device)
    conv.weight.data.fill_(1.0 / 3.0)
    conv.requires_grad_(False)

    conv_grad = conv(grad_x + 0.5 * grad2_x)

    # Edge weights
    edge_weights = 1.0 + edge_factor * conv_grad
    total_weights = amp_weights * edge_weights
    weighted_loss = (total_weights * mse_loss).mean()

    return weighted_loss



# Time step：focus more on the early time steps
def sample_timesteps(batch_size, device, strategy='importance'):
    if strategy == 'importance':

        probs = np.exp(-np.linspace(0, 3, dc['T']))
        probs = probs / probs.sum()
        steps = np.random.choice(dc['T'], size=batch_size, p=probs)
        return torch.tensor(steps, device=device).long()
    else:
        # uniform sampling
        return torch.randint(0, dc['T'], (batch_size,), device=device).long()

#%% md
# Denoisy Process, Inference Phase
#%%

# DDPM Function (Denoisy Process, Inference Phase)
def multi_step_denoise(model, dc, noisy_tensor, device, num_steps=1000):
    model.eval()
    with torch.no_grad():
        x_t = noisy_tensor.to(device)

        if isinstance(dc['timesteps'], (np.ndarray, list)):
            timesteps_num = int(dc['timesteps'][0])
        else:
            timesteps_num = int(dc['timesteps'])

        timesteps = torch.linspace(timesteps_num - 1, 0, num_steps).long().to(device)

        for t in timesteps:
            step_batch = torch.tensor([t], dtype=torch.long, device=device)
            pred_noise, _ = model(x_t, step_batch)

            # DDPM theoretical formula
            alpha_t = torch.tensor(dc['alphas'][t], dtype=torch.float32, device=device)
            alpha_bar_t = torch.tensor(dc['alphas_bar'][t], dtype=torch.float32, device=device)
            beta_t = torch.tensor(dc['betas'][t], dtype=torch.float32, device=device)

            x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise
            else:
                x_t = x0_pred

        return x_t.cpu()

#%% md
# Transition point identity
#%%
#identiy threshold by file name
def get_threshold_from_path(file_path):
    folder_name = os.path.basename(os.path.dirname(file_path))

    if '2-state' in folder_name.lower():
        return 0.5
    elif '3-state' in folder_name.lower():
        return [0.25, 0.75]
    elif '4-state' in folder_name.lower():
        return [0.165, 0.5, 0.83]



#Transition point identity function
def detect_transition_points_with_state(signal, threshold, min_state_len=10):
    transition_points = []
    transition_states = []

    if not isinstance(threshold, list):
        def get_state(val):
            if isinstance(val, (np.ndarray, torch.Tensor)):
                return 1 if (val > threshold).any() else 0
            else:
                return 1 if val > threshold else 0
    else:
        def get_state(val):
            state = 0
            for t in threshold:
                if isinstance(val, (np.ndarray, torch.Tensor)):
                    if (val > t).any():
                        state += 1
                else:
                    if val > t:
                        state += 1
            return state

    states = [get_state(val) for val in signal]

    prev_state = states[0]
    segment_start = 0

    for i in range(1, len(states)):
        if states[i] != prev_state:
            segment_len = i - segment_start
            if segment_len >= min_state_len:
                transition_points.append(i)
                transition_states.append(states[i])
                segment_start = i
                prev_state = states[i]

    return np.array(transition_points), np.array(transition_states)


# Transition point matching
def improved_transition_point_match(gt_points, pred_points, window_size=2):
    matches = []
    used_pred = set()

    for i, gt_pos in enumerate(gt_points):
        best_match = None
        min_distance = float('inf')

        for j, pred_pos in enumerate(pred_points):
            if j in used_pred:
                continue

            distance = abs(pred_pos - gt_pos)
            if distance <= window_size and distance < min_distance:
                min_distance = distance
                best_match = j

        if best_match is not None:
            matches.append((i, best_match))
            used_pred.add(best_match)

    return {
        'matches': [(gt_points[i], pred_points[j]) for i, j in matches],
        'unmatched_gt': [gt_points[i] for i in range(len(gt_points))
                         if i not in [m[0] for m in matches]],
        'unmatched_pred': [pred_points[j] for j in range(len(pred_points))
                           if j not in used_pred]
    }


class TransitionPointEvaluator:
    def __init__(self, tolerance=2):
        self.tolerance = tolerance
        self.all_results = []

    def evaluate_transition_points(self, gt_points, pred_points, signal_gt, signal_pred):
        match_result = improved_transition_point_match(gt_points, pred_points, self.tolerance)

        matches = match_result['matches']
        unmatched_gt = match_result['unmatched_gt']
        unmatched_pred = match_result['unmatched_pred']

        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if matches:
            time_errors = [abs(gt_pos - pred_pos) for gt_pos, pred_pos in matches]
            avg_time_error = np.mean(time_errors)
            value_errors = [abs(signal_gt[gt_pos] - signal_pred[pred_pos])
                            for gt_pos, pred_pos in matches]
            avg_value_error = np.mean(value_errors)
        else:
            avg_time_error = float('inf') if len(pred_points) > 0 else 0
            avg_value_error = float('inf') if len(pred_points) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'avg_time_error': avg_time_error,
            'avg_value_error': avg_value_error,
            'matches': matches,
            'unmatched_gt': unmatched_gt,
            'unmatched_pred': unmatched_pred
        }

    def add_sample_result(self, gt_transition_points, denoised_transition_points,
                          gt_signal, denoised_signal):
        result = self.evaluate_transition_points(gt_transition_points, denoised_transition_points,
                                             gt_signal, denoised_signal)
        self.all_results.append(result)
        return result

    def summarize_results(self):
        if not self.all_results:
            return None

        total_tp = sum(res['tp'] for res in self.all_results)
        total_fp = sum(res['fp'] for res in self.all_results)
        total_fn = sum(res['fn'] for res in self.all_results)

        precision_list = [res['precision'] for res in self.all_results]
        recall_list = [res['recall'] for res in self.all_results]
        f1_list = [res['f1'] for res in self.all_results]

        time_error_list = [res['avg_time_error'] for res in self.all_results
                           if res['avg_time_error'] != float('inf')]
        value_error_list = [res['avg_value_error'] for res in self.all_results
                            if res['avg_value_error'] != float('inf')]

        return {
            'num_samples': len(self.all_results),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'performance': {
                'avg_precision': np.mean(precision_list),
                'avg_recall': np.mean(recall_list),
                'avg_f1': np.mean(f1_list),
                'avg_time_error': np.mean(time_error_list) if time_error_list else float('inf'),
                'std_time_error': np.std(time_error_list) if time_error_list else 0,
                'avg_value_error': np.mean(value_error_list) if value_error_list else float('inf'),
                'std_value_error': np.std(value_error_list) if value_error_list else 0,
                'samples_with_valid_matches': len(time_error_list)
            }
        }


# Evaluation for transition points
def evaluate_model(model, dc, test_dataset, device, num_steps=1000, visualize=False):
    mse_list = []
    score_total_list = []

    evaluator = TransitionPointEvaluator(tolerance=2)
    folder_results = defaultdict(list)
    folder_mse = defaultdict(list)

    for sample_idx in range(len(test_dataset)):
        sample = test_dataset[sample_idx]
        if len(sample) == 3:
            noisy_tensor, clean_tensor, path = sample
        else:
            noisy_tensor, clean_tensor = sample
            path = None


        if noisy_tensor.dim() == 1:
            noisy_tensor_input = noisy_tensor.unsqueeze(0).unsqueeze(0).to(device)
            clean_tensor_input = clean_tensor.unsqueeze(0).unsqueeze(0).to(device)
        elif noisy_tensor.dim() == 2:
            noisy_tensor_input = noisy_tensor.unsqueeze(0).to(device)
            clean_tensor_input = clean_tensor.unsqueeze(0).to(device)
        else:
            noisy_tensor_input = noisy_tensor.to(device)
            clean_tensor_input = clean_tensor.to(device)


        denoised = multi_step_denoise(model, dc, noisy_tensor_input, device, num_steps=num_steps)



        gt_signal = clean_tensor_input.squeeze().cpu().numpy()
        denoised_signal = denoised.squeeze().cpu().numpy()


        if path:
            threshold = get_threshold_from_path(path)
        else:
            threshold = 0.5

        gt_transition_points, _ = detect_transition_points_with_state(gt_signal, threshold, min_state_len=10)
        denoised_transition_points, _ = detect_transition_points_with_state(denoised_signal, threshold, min_state_len=10)

        # MSE calculation
        mse = np.mean((denoised_signal - gt_signal) ** 2)
        mse_list.append(mse)


        result = evaluator.add_sample_result(gt_transition_points, denoised_transition_points,
                                             gt_signal, denoised_signal)

        if path:
            folder_name = os.path.basename(os.path.dirname(path))
            folder_results[folder_name].append(result)
            folder_mse[folder_name].append(mse)


    summary = evaluator.summarize_results()

    # Calculate average metrics by folder
    print("\n Folder Results")
    for folder, results in folder_results.items():
        if not results:
            continue
        precision = np.mean([r['precision'] for r in results])
        recall = np.mean([r['recall'] for r in results])
        f1 = np.mean([r['f1'] for r in results])
        time_err = np.mean([r['avg_time_error'] for r in results if r['avg_time_error'] != float('inf')])
        mse_avg = np.mean(folder_mse[folder]) if folder_mse[folder] else float('inf')
        score_total = np.log(max(f1 / mse_avg, 1e-8))
        print(f"Folder: {folder:20s} |F1={f1:.6f}, MSE={mse_avg:.6f}, Score={score_total:.4=6f}")

    # Score calculation
    for mse in mse_list:
        f1 = summary['performance']['avg_f1'] if summary else 0
        ratio = f1 / mse
        score_total = np.log(max(ratio, 1e-8))

        score_total_list.append(score_total)

    return np.mean(mse_list), np.mean(score_total_list), summary, mse_list
#%% md
# Hyperparameter
#%%
N_BASE_CHANNELS = 192
LR = 0.000076137
BATCH_SIZE = 16
EPOCHS = 150
AMP_FACTOR = 14.52647
EDGE_FACTOR = 8.94621
#%% md
# Training process
#%%
if __name__ == "__main__":

    '''Note: please modify the train_dataset path and test_dataset path accordingly'''

    # train dataset
    train_dataset = StepwiseDataset("./data/train",selected_ids=list(range(1, 100)))
    # test dataset
    test_dataset = StepwiseDataset("./data/train", selected_ids=list(range(1, 21)))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Build model
    model = DiffusionUNetLegacy(
        name='unet',
        dims=1,
        n_in_channels=1,
        n_base_channels=N_BASE_CHANNELS,
        n_emb_dim=256,
        n_enc_blocks=4,
        n_dec_blocks=4,
        n_groups=16,
        n_heads=4,
        actv=nn.SiLU(),
        kernel_size=3,
        padding=1,
        use_attention=True,
        skip_connection=True,
        chnnel_multiples=(1,2,4,8),
        updown_rates=(1,2,2,2),
        use_scale_shift_norm=True,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        for noisy_batch, clean_batch, _ in train_loader:
            x_0_batch = clean_batch.to(device)
            step_batch = sample_timesteps(x_0_batch.size(0), device)

            x_t_batch, noise = forward_sample(x_0_batch, step_batch, dc)
            noise_pred, _ = model(x_t_batch, step_batch)

            loss = compute_enhanced_loss(
                noise_pred, noise, x_0_batch,
                AMP_FACTOR, EDGE_FACTOR
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss={np.mean(epoch_losses):.6f}")


    #  Evaluation
    train_mse, train_score, train_summary, _ = evaluate_model(model, dc, train_dataset, device)

    test_mse, test_score, test_summary, _ = evaluate_model(model, dc, test_dataset, device)

    print(f"Train MSE: {train_mse:.6f}")
    if train_summary:
        print(f"Train F1: {train_summary['performance']['avg_f1']:.6f}")
        print(f"Train Score: {train_score:.6f}")

    print(f"\nTest MSE: {test_mse:.6f}")
    if test_summary:
        print(f"Test F1: {test_summary['performance']['avg_f1']:.6f}")
        print(f"Test Score: {test_score:.6f}")
