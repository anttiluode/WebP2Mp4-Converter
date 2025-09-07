# MorphogeneticImageStudio.py
# An advanced VAE trainer for images, featuring biologically-inspired principles of
# damage/repair and ephaptic field dynamics for studying computational morphogenesis.
# 
# Based on the concepts from the research of Michael Levin, this tool allows you to:
#  1. Train a VAE on any folder of images.
#  2. Apply targeted "damage" to the network's weights during training.
#  3. Observe the network's self-repair process through a live "Field Dynamics" dashboard.
#  4. Utilize ephaptic coupling (non-local field effects) to guide this repair process.

import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import gc
import json
import random
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------
# Ephaptic Coupling & Field Analysis Modules
# (Adapted from llm_studio4.py and forms3.py)
# ---------------------------------------------------

class EphapticCoupling:
    """Biologically-inspired ephaptic coupling to provide non-local field effects."""
    def __init__(self, coupling_strength=0.01, spatial_scale=5.0, temporal_decay=0.95):
        self.coupling_strength = coupling_strength
        self.spatial_scale = spatial_scale
        self.temporal_decay = temporal_decay
        self.field_states = {} # Stores field history for temporal effects

    def create_smoothing_kernel(self, channels, device):
        """Creates a Gaussian kernel whose properties are modulated by the field strength."""
        if self.coupling_strength <= 1e-6:
            return None
        
        sigma = 0.5 + 2.5 * self.coupling_strength
        kernel_size = 7
        ax = torch.linspace(-(kernel_size//2), kernel_size//2, steps=kernel_size, device=device)
        kernel1d = torch.exp(-0.5 * (ax / sigma)**2)
        kernel1d /= kernel1d.sum()
        kernel2d = torch.outer(kernel1d, kernel1d)
        
        # Make it a depthwise kernel
        kernel = kernel2d[None, None, :, :].repeat(channels, 1, 1, 1)
        return kernel

    def apply_field_smoothing(self, feature_map, device):
        """Apply the ephaptic field effect as a global smoothing operation."""
        kernel = self.create_smoothing_kernel(feature_map.shape[1], device)
        if kernel is None:
            return feature_map
        
        padding = kernel.shape[-1] // 2
        return F.conv2d(feature_map, kernel, padding=padding, groups=feature_map.shape[1])

def extract_weight_field(model):
    """Extracts all model weights and reshapes them into a 2D square 'field'."""
    with torch.no_grad():
        params = [p.flatten() for p in model.parameters()]
        all_params = torch.cat(params).cpu().numpy()
    
    n = len(all_params)
    side = int(np.ceil(np.sqrt(n)))
    field = np.zeros(side * side)
    field[:n] = all_params
    return field.reshape(side, side)

def polar_energy(mag, bins_theta=72, eps=1e-9):
    H, W = mag.shape
    cy, cx = (H-1)/2, (W-1)/2
    ys, xs = np.indices((H, W))
    ang = np.arctan2(ys-cy, xs-cx)
    ang = (ang + np.pi) % np.pi
    theta_edges = np.linspace(0, np.pi, bins_theta+1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    E = np.zeros(bins_theta, dtype=np.float64)
    for i in range(bins_theta):
        mask = (ang >= theta_edges[i]) & (ang < theta_edges[i+1])
        E[i] = mag[mask].sum()
    E /= (E.sum() + eps)
    return theta_centers, E

def orientation_selectivity(E, theta):
    """Calculates the Orientation Selectivity Index (OSI) from the energy histogram."""
    vx = np.sum(E * np.cos(2*theta))
    vy = np.sum(E * np.sin(2*theta))
    return float(np.hypot(vx, vy))

def compute_field_coherence_and_orientation(field1, field2):
    """Computes coherence and OSI from the change between two weight fields."""
    delta = field2 - field1
    magnitude = np.abs(np.fft.fft2(delta))
    mag_norm = magnitude / (magnitude.sum() + 1e-8)
    
    entropy = -np.sum(mag_norm * np.log(mag_norm + 1e-8))
    coherence = 1.0 - (entropy / np.log(mag_norm.size))
    
    theta, E = polar_energy(magnitude)
    osi = orientation_selectivity(E, theta)
    
    return coherence, magnitude, osi, theta, E

def apply_network_damage(model, damage_fraction=0.2):
    """Zeros out a fraction of weights in the model to simulate injury."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                mask = torch.rand_like(param) < damage_fraction
                param[mask] = 0.0
    print(f"Applied {damage_fraction*100:.1f}% damage to network weights")

# -----------------------------
# Shape VAE Model
# -----------------------------

class ImageVAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=64, ephaptic_config=None):
        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Deconvolutional decoder
        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
        
        # Ephaptic Coupling module
        if ephaptic_config:
            self.ephaptic = EphapticCoupling(**ephaptic_config)
        else:
            self.ephaptic = None

    def set_field_strength(self, strength):
        if self.ephaptic:
            self.ephaptic.coupling_strength = strength

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, device):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 4, 4)
        
        # Apply ephaptic smoothing here in the decoder's generative process
        if self.ephaptic and self.training:
            h = self.ephaptic.apply_field_smoothing(h, device)
            
        return self.decoder_conv(h)

    def forward(self, x, device):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, device)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, num_samples=1, device='cpu'):
        self.eval()
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z, device)

# -----------------------------
# Dataset and Training
# -----------------------------

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Warning: Skipping corrupted image {self.image_paths[idx]}: {e}")
            return None # Dataloader will ignore this

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.Tensor()
    return torch.stack(batch, 0)

def train_epoch_with_fields(model, dataloader, optimizer, device, visualizer, log_fn):
    model.train()
    total_loss_sum = 0
    batch_count = 0
    step_counter = visualizer.get_step_counter()
    
    prev_field = extract_weight_field(model)

    for i, images in enumerate(dataloader):
        if images.nelement() == 0: continue
        images = images.to(device)
        
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images, device)
        
        recon_loss = F.mse_loss(recon_images, images, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + 0.001 * kl_loss) / images.size(0)
        
        loss.backward()
        optimizer.step()
        
        total_loss_sum += loss.item()
        batch_count += 1
        step_counter += 1
        
        # Update field dynamics and visualization every few batches
        if i % 10 == 0:
            with torch.no_grad():
                current_field = extract_weight_field(model)
                coherence, osi = visualizer.update(
                    model, images, recon_images, loss.item(), step_counter
                )
                
                # Feedback loop: Coherence and OSI modulate ephaptic strength
                base_strength = 0.2 * coherence
                osi_boost = 0.05 * osi
                new_field_strength = np.clip(base_strength + osi_boost, 0.0, 0.4)
                model.set_field_strength(new_field_strength)
                
                prev_field = current_field
                
                if log_fn:
                    log_fn(f"Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f} | "
                           f"Coherence: {coherence:.3f} | OSI: {osi:.3f} | Field Str: {model.ephaptic.coupling_strength:.3f}\n")
    
    visualizer.set_step_counter(step_counter)
    return total_loss_sum / batch_count if batch_count > 0 else 0

# -------------------------------------------
# Live Visualization Dashboard
# -------------------------------------------

class FieldDynamicsVisualizer:
    def __init__(self, fig):
        self.fig = fig
        gs = self.fig.add_gridspec(3, 4)
        self.ax_input = self.fig.add_subplot(gs[0, 0])
        self.ax_output = self.fig.add_subplot(gs[0, 1])
        self.ax_weights = self.fig.add_subplot(gs[0, 2])
        self.ax_fft = self.fig.add_subplot(gs[0, 3])
        self.ax_evo = self.fig.add_subplot(gs[1, :])
        self.ax_field_strength = self.fig.add_subplot(gs[2, :2])
        self.ax_orientation = self.fig.add_subplot(gs[2, 2:])
        
        self.history = {'loss': [], 'coherence': [], 'osi': [], 'field': []}
        self.prev_field = None
        self.step_counter = 0
        self.damage_events = []

    def get_step_counter(self): return self.step_counter
    def set_step_counter(self, val): self.step_counter = val
    def mark_damage_event(self): self.damage_events.append(self.step_counter)

    def update(self, model, sample_in, sample_out, loss, step):
        current_field = extract_weight_field(model)
        coherence, osi, theta, E = 0.0, 0.0, np.array([]), np.array([])
        
        if self.prev_field is not None:
            coherence, fft_mag, osi, theta, E = compute_field_coherence_and_orientation(self.prev_field, current_field)
            fft_log = np.log1p(np.fft.fftshift(fft_mag))
            self.ax_fft.imshow(fft_log, cmap='hot', aspect='auto')
        
        self.history['loss'].append(loss)
        self.history['coherence'].append(coherence)
        self.history['osi'].append(osi)
        self.history['field'].append(model.ephaptic.coupling_strength if model.ephaptic else 0)
        
        self.prev_field = current_field
        
        # Update plots
        self.ax_input.imshow(sample_in[0].cpu().permute(1,2,0).numpy())
        self.ax_output.imshow(sample_out[0].cpu().permute(1,2,0).numpy())
        self.ax_weights.imshow(current_field, cmap='viridis', aspect='auto')
        self.ax_orientation.clear()
        self.ax_orientation.plot(theta * 180 / np.pi, E)
        self.ax_orientation.set_title(f'Orientation Energy (OSI: {osi:.3f})')

        # Combined Evolution Plot
        self.ax_evo.clear()
        ax1 = self.ax_evo
        ax2 = ax1.twinx()
        ax1.plot(self.history['loss'], 'r-', label='Loss', alpha=0.7)
        ax2.plot(self.history['coherence'], 'b-', label='Coherence')
        ax2.plot(self.history['osi'], 'g-', label='OSI')
        ax1.set_ylabel('Loss', color='r')
        ax2.set_ylabel('Coherence/OSI', color='b')
        ax1.set_title('Training Evolution')
        
        # Field Strength Plot
        self.ax_field_strength.clear()
        self.ax_field_strength.plot(self.history['field'], 'purple')
        self.ax_field_strength.set_title('Ephaptic Field Strength')
        
        for event in self.damage_events:
            ax1.axvline(event, color='orange', linestyle='--', label='Damage')
            
        self.fig.canvas.draw()
        plt.pause(0.01) # Allow GUI to update
        
        return coherence, osi

# -----------------------------
# Main GUI Application
# -----------------------------

class MorphogeneticImageStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Morphogenetic Image Studio")
        self.geometry("1400x1000")

        self.model = None
        self.image_paths = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training params
        self.batch_size = tk.IntVar(value=16)
        self.epochs = tk.IntVar(value=20)
        self.lr = tk.DoubleVar(value=1e-4)
        self.latent_dim = tk.IntVar(value=64)
        
        # Ephaptic params
        self.use_ephaptic = tk.BooleanVar(value=True)
        self.coupling_strength = tk.DoubleVar(value=0.01)
        self.spatial_scale = tk.DoubleVar(value=5.0)
        self.temporal_decay = tk.DoubleVar(value=0.95)
        
        self._build_ui()
        self._check_gpu_memory()

    def _build_ui(self):
        left_panel = tk.Frame(self, width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)
        right_frame = tk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- LEFT PANEL ---
        
        # Dataset Management
        dataset_frame = tk.LabelFrame(left_panel, text="Dataset Management")
        dataset_frame.pack(fill=tk.X, pady=5)
        tk.Button(dataset_frame, text="Load Images From Folder...", command=self._load_images, bg="#d3e3f3", font=("Arial", 10, "bold")).pack(pady=5, fill=tk.X)
        self.dataset_status_label = tk.Label(dataset_frame, text="Dataset: 0 images", relief=tk.SUNKEN)
        self.dataset_status_label.pack(fill=tk.X, pady=5)
        tk.Button(dataset_frame, text="Clear Dataset", command=self._clear_dataset, bg="#f5dcdc").pack(pady=5, fill=tk.X)

        # Model Configuration
        model_frame = tk.LabelFrame(left_panel, text="Model Configuration")
        model_frame.pack(fill=tk.X, pady=5)
        tk.Label(model_frame, text="Latent Dimension:").pack(anchor=tk.W)
        tk.Entry(model_frame, textvariable=self.latent_dim).pack(anchor=tk.W)
        tk.Button(model_frame, text="Initialize Model", command=self._init_model, bg="#d3f3e3", font=("Arial", 10, "bold")).pack(pady=10, fill=tk.X)

        # Ephaptic Coupling
        ephaptic_frame = tk.LabelFrame(left_panel, text="Ephaptic Coupling")
        ephaptic_frame.pack(fill=tk.X, pady=5)
        tk.Checkbutton(ephaptic_frame, text="Enable Ephaptic Fields", variable=self.use_ephaptic).pack(anchor=tk.W)
        tk.Label(ephaptic_frame, text="Initial Strength:").pack(anchor=tk.W)
        tk.Entry(ephaptic_frame, textvariable=self.coupling_strength).pack(anchor=tk.W)

        # Training
        train_frame = tk.LabelFrame(left_panel, text="Training Controls")
        train_frame.pack(fill=tk.X, pady=5)
        tk.Label(train_frame, text="Batch Size:").pack(anchor=tk.W); tk.Entry(train_frame, textvariable=self.batch_size).pack(anchor=tk.W)
        tk.Label(train_frame, text="Epochs:").pack(anchor=tk.W); tk.Entry(train_frame, textvariable=self.epochs).pack(anchor=tk.W)
        tk.Label(train_frame, text="Learning Rate:").pack(anchor=tk.W); tk.Entry(train_frame, textvariable=self.lr).pack(anchor=tk.W)
        self.train_btn = tk.Button(train_frame, text="Start Training", command=self._start_training, bg="#90EE90", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.train_btn.pack(pady=10, fill=tk.X)

        # Damage & Repair
        damage_frame = tk.LabelFrame(left_panel, text="Damage & Repair")
        damage_frame.pack(fill=tk.X, pady=5)
        self.damage_fraction = tk.DoubleVar(value=0.2)
        tk.Label(damage_frame, text="Damage Fraction (0-1):").pack(anchor=tk.W)
        tk.Entry(damage_frame, textvariable=self.damage_fraction).pack(anchor=tk.W)
        self.damage_btn = tk.Button(damage_frame, text="Apply Damage Now", command=self._apply_damage, bg="#ffcccc", state=tk.DISABLED)
        self.damage_btn.pack(pady=5, fill=tk.X)
        
        # Model I/O
        io_frame = tk.LabelFrame(left_panel, text="Model I/O")
        io_frame.pack(fill=tk.X, pady=5)
        self.save_btn = tk.Button(io_frame, text="Save Model", command=self._save_model, state=tk.DISABLED)
        self.save_btn.pack(pady=2, fill=tk.X)
        tk.Button(io_frame, text="Load Model", command=self._load_model).pack(pady=2, fill=tk.X)
        
        # Status
        self.gpu_label = tk.Label(left_panel, text="GPU: N/A", bg="#f0f0f0", relief=tk.SUNKEN)
        self.gpu_label.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(left_panel, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # --- RIGHT FRAME (NOTEBOOK) ---
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Training Log")
        self.log = scrolledtext.ScrolledText(log_frame, height=10, font=('Courier', 9))
        self.log.pack(fill=tk.BOTH, expand=True)

        # Field Dynamics Tab
        field_tab = ttk.Frame(notebook)
        notebook.add(field_tab, text="Field Dynamics Dashboard")
        self.field_fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.field_canvas = FigureCanvasTkAgg(self.field_fig, master=field_tab)
        self.field_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.visualizer = FieldDynamicsVisualizer(self.field_fig)

    # --- Backend Functions ---

    def _log(self, msg):
        self.log.insert(tk.END, msg)
        self.log.see(tk.END)
        self.update_idletasks()

    def _check_gpu_memory(self):
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            alloc = torch.cuda.memory_allocated(0) / 1e9
            free = total - alloc
            self.gpu_label.config(text=f"GPU: {alloc:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")
        else:
            self.gpu_label.config(text="GPU: CUDA not available", bg="#ffcccc")
            
    def _load_images(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path: return
        
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.dataset_status_label.config(text=f"Dataset: {len(self.image_paths)} images")
        self._log(f"Loaded {len(self.image_paths)} image paths.\n")
        if self.model: self.train_btn.config(state=tk.NORMAL)
        
    def _clear_dataset(self):
        self.image_paths = []
        self.dataset_status_label.config(text="Dataset: 0 images")
        self.train_btn.config(state=tk.DISABLED)

    def _init_model(self):
        try:
            ephaptic_config = {
                'coupling_strength': self.coupling_strength.get(),
                'spatial_scale': self.spatial_scale.get(),
                'temporal_decay': self.temporal_decay.get()
            } if self.use_ephaptic.get() else None
            
            self.model = ImageVAE(latent_dim=self.latent_dim.get(), ephaptic_config=ephaptic_config).to(self.device)
            params = sum(p.numel() for p in self.model.parameters()) / 1e6
            self._log(f"Model initialized: {params:.2f}M parameters on {self.device}\n")
            if self.image_paths: self.train_btn.config(state=tk.NORMAL)
            self.damage_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Model init failed: {e}")
            
    def _apply_damage(self):
        if not self.model: return
        apply_network_damage(self.model, self.damage_fraction.get())
        self.visualizer.mark_damage_event()
        self._log(f"Applied {self.damage_fraction.get()*100:.1f}% damage.\n")
        
    def _start_training(self):
        if not self.model or not self.image_paths:
            messagebox.showwarning("Warning", "Initialize model and load data first.")
            return
        
        self.train_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        try:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            dataset = CustomImageDataset(self.image_paths, transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size.get(), shuffle=True, collate_fn=custom_collate, num_workers=0)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr.get())
            
            self._log("Starting training...\n")
            for epoch in range(self.epochs.get()):
                self._log(f"--- Epoch {epoch+1}/{self.epochs.get()} ---\n")
                avg_loss = train_epoch_with_fields(self.model, dataloader, optimizer, self.device, self.visualizer, self._log)
                self._log(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}\n")
                self._check_gpu_memory()
            
            self._log("Training finished.\n")
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self._log(f"Training error: {e}\n")
            import traceback
            self._log(traceback.format_exc() + "\n")
        finally:
            self.progress.stop()
            self.train_btn.config(state=tk.NORMAL)
            
    def _save_model(self):
        if not self.model: return
        path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch models", "*.pt")])
        if path:
            save_obj = {
                'model_state_dict': self.model.state_dict(),
                'latent_dim': self.latent_dim.get(),
                'ephaptic_config': self.model.ephaptic.__dict__ if self.model.ephaptic else None
            }
            torch.save(save_obj, path)
            self._log(f"Model saved to {path}\n")

    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch models", "*.pt")])
        if path:
            checkpoint = torch.load(path, map_location=self.device)
            self.latent_dim.set(checkpoint['latent_dim'])
            
            # Re-init model with loaded config
            self._init_model() 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self._log(f"Model loaded from {path}\n")
            self.save_btn.config(state=tk.NORMAL)
            self.damage_btn.config(state=tk.NORMAL)
            if self.image_paths: self.train_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    app = MorphogeneticImageStudio()
    app.mainloop()