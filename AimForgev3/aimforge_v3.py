#!/usr/bin/env python3
import sys
import subprocess

# Auto‐install missing dependencies
required = ['numpy', 'torch', 'pynput', 'pywin32', 'win10toast']
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

import os
import time
import threading
import glob
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from pynput import mouse, keyboard
import win32gui
from win10toast import ToastNotifier

# --- Directories and constants ---

DATA_FILE    = 'mouse_data.csv'
MODELS_DIR   = 'models'
LOGS_DIR     = 'logs'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# --- Neural network definition ---

class AimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

# --- Pipeline functions with notifications and stop events ---

def record_data(duration=None, app_filter='', log_cb=print,
                notify=None, stop_event=None):
    """
    Records raw mouse deltas plus app hash into DATA_FILE.
    If app_filter is non‐empty, only records when the foreground
    window title contains app_filter.
    """
    prev_moves = [(0, 0), (0, 0)]
    last_pos = [None, None]
    start_time = time.time()

    def get_app_info():
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd) or 'Unknown'
        h = abs(hash(title)) % 1_000_000
        return h, title

    log_cb(f"Recording to '{DATA_FILE}' "
           f"{'(filtered by '+app_filter+')' if app_filter else ''}")
    if notify:
        notify("AimAssist", "Recording started", duration=3)

    def on_move(x, y):
        # stop if requested or duration elapsed
        if stop_event and stop_event.is_set():
            return False
        if duration and (time.time() - start_time) > duration:
            return False

        if last_pos[0] is None:
            dx = dy = 0
        else:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
        last_pos[0], last_pos[1] = x, y

        h, title = get_app_info()
        if app_filter and app_filter.lower() not in title.lower():
            return  # skip if not matching filter

        f1, f2 = prev_moves
        row = [
            time.time(),
            f1[0], f1[1], f2[0], f2[1],
            h, dx, dy
        ]
        with open(DATA_FILE, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        prev_moves.pop(0)
        prev_moves.append((dx, dy))

    try:
        with mouse.Listener(on_move=on_move) as listener:
            listener.join()
    except Exception as e:
        log_cb(f"Recording stopped: {e}")
    else:
        log_cb("Recording finished")
    finally:
        if notify:
            notify("AimAssist", "Recording stopped", duration=3)

def train_model(epochs=30, batch_size=256, lr=1e-3,
                model_name=None, target_app=None,
                log_cb=print, notify=None, stop_event=None):
    """
    Trains an AimNet on DATA_FILE and saves the checkpoint
    under models/<model_name>.pth including normalization stats
    and target_app metadata.
    """
    if not model_name:
        raise ValueError("Model name must be provided")
    data_path = DATA_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file '{data_path}' not found")

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    log_cb(f"Training '{model_name}' for '{target_app}'")
    if notify:
        notify("AimAssist", f"Training '{model_name}' started", duration=3)

    # Load and preprocess
    data = np.loadtxt(data_path, delimiter=',')
    X, y = data[:,1:6], data[:,6:8]
    Xm, Xs = X.mean(0), X.std(0)+1e-6
    ym, ys = y.mean(0), y.std(0)+1e-6
    Xn, yn = (X-Xm)/Xs, (y-ym)/ys

    # Split
    n = len(Xn)
    idx = np.random.permutation(n)
    split = int(0.8*n)
    t_idx, v_idx = idx[:split], idx[split:]
    train_ds = TensorDataset(torch.from_numpy(Xn[t_idx]).float(),
                             torch.from_numpy(yn[t_idx]).float())
    val_ds   = TensorDataset(torch.from_numpy(Xn[v_idx]).float(),
                             torch.from_numpy(yn[v_idx]).float())
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size)

    model = AimNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(1, epochs+1):
        if stop_event and stop_event.is_set():
            log_cb("Training aborted by user")
            break

        model.train()
        for xb, yb in train_ld:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            if stop_event and stop_event.is_set():
                break

        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_ld:
                losses.append(criterion(model(xb), yb).item())
        val_loss = float(sum(losses)/len(losses))
        log_cb(f"Epoch {epoch}/{epochs} — Val MSE: {val_loss:.6f}")

    # Save checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'X_mean': Xm, 'X_std': Xs,
        'y_mean': ym, 'y_std': ys,
        'app_name': target_app
    }, model_path)
    log_cb(f"Training complete, saved to '{model_path}'")
    if notify:
        notify("AimAssist", f"Training '{model_name}' complete", duration=3)

def run_assist(model_path, alpha=0.3, target_app='',
               toggle_key='F8', log_cb=print,
               notify=None, stop_event=None):
    """
    Loads the checkpoint and injects aim corrections in real time
    when the foreground window title contains target_app.
    Allows toggling assist on/off with toggle_key.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' not found")

    ckpt = torch.load(model_path)
    model = AimNet()
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    Xm, Xs = ckpt['X_mean'], ckpt['X_std']
    ym, ys = ckpt['y_mean'], ckpt['y_std']

    prev_moves = [(0, 0), (0, 0)]
    last_pos = [None, None]
    assist_on = True

    # Set up keyboard toggle
    if toggle_key.startswith('F'):
        key_enum = getattr(keyboard.Key, toggle_key.lower(), None)
        char_match = None
    else:
        key_enum = None
        char_match = toggle_key.upper()

    def on_key_press(key):
        nonlocal assist_on
        hit = False
        if key_enum and key == key_enum:
            hit = True
        elif char_match and hasattr(key, 'char') and key.char and key.char.upper() == char_match:
            hit = True
        if hit:
            assist_on = not assist_on
            state = "ON" if assist_on else "OFF"
            log_cb(f"Assist toggled {state}")
            if notify:
                notify("AimAssist", f"Assist {state}", duration=2)

    kb_listener = keyboard.Listener(on_press=on_key_press)
    kb_listener.start()

    def get_app_info():
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd) or 'Unknown'
        h = abs(hash(title)) % 1_000_000
        return h, title

    log_cb(f"Assist running ({'enabled' if assist_on else 'disabled'}) "
           f"for '{target_app}', toggle key: {toggle_key}")
    if notify:
        notify("AimAssist", "Assist started", duration=3)

    def on_move(x, y):
        if stop_event and stop_event.is_set():
            return False

        if last_pos[0] is None:
            dx = dy = 0
        else:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
        last_pos[0], last_pos[1] = x, y

        h, title = get_app_info()
        if target_app and target_app.lower() not in title.lower():
            return  # skip if not matching

        raw = np.array([
            prev_moves[0][0], prev_moves[0][1],
            prev_moves[1][0], prev_moves[1][1],
            h
        ], dtype=float)
        feat = (raw - Xm) / Xs
        with torch.no_grad():
            predn = model(torch.from_numpy(feat).float())
        adj = (predn.numpy() * ys) + ym

        if assist_on:
            mouse.Controller().move(alpha * adj[0], alpha * adj[1])

        prev_moves.pop(0)
        prev_moves.append((dx, dy))

    try:
        with mouse.Listener(on_move=on_move) as m_listener:
            m_listener.join()
    except Exception as e:
        log_cb(f"Assist stopped: {e}")
    finally:
        kb_listener.stop()
        if notify:
            notify("AimAssist", "Assist stopped", duration=3)

# --- Tkinter GUI application ---

class AimAssistGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AimForge")
        self.geometry("650x480")
        self.resizable(False, False)

        # Notification toaster
        self.toaster = ToastNotifier()

        # Prepare a log file for this session
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOGS_DIR, f"log_{ts}.txt")
        self.log_file = open(self.log_path, 'a', encoding='utf-8')

        # Thread stop events
        self.record_event = threading.Event()
        self.train_event  = threading.Event()
        self.assist_event = threading.Event()

        self._build_gui()
        self._update_status("Idle")

    def _build_gui(self):
        nb = ttk.Notebook(self)
        # Record tab
        rec = ttk.Frame(nb); nb.add(rec, text="Record")
        ttk.Label(rec, text="Duration (sec, empty=until stopped):")\
            .grid(row=0, column=0, pady=5, sticky="w")
        self.rec_dur = ttk.Entry(rec, width=10); self.rec_dur.grid(row=0, column=1)
        ttk.Label(rec, text="App filter (window title):")\
            .grid(row=1, column=0, pady=5, sticky="w")
        self.rec_filter = ttk.Entry(rec, width=20); self.rec_filter.grid(row=1, column=1)
        self.rec_start = ttk.Button(rec, text="Start", command=self._start_record)
        self.rec_start.grid(row=2, column=0, pady=10)
        self.rec_stop  = ttk.Button(rec, text="Stop",  command=self._stop_record,
                                    state="disabled")
        self.rec_stop.grid(row=2, column=1)

        # Train tab
        trn = ttk.Frame(nb); nb.add(trn, text="Train")
        ttk.Label(trn, text="Model name:")\
            .grid(row=0, column=0, sticky="w")
        self.trn_name = ttk.Entry(trn, width=20); self.trn_name.grid(row=0, column=1)
        ttk.Label(trn, text="Target app:")\
            .grid(row=1, column=0, sticky="w")
        self.trn_app  = ttk.Entry(trn, width=20); self.trn_app.grid(row=1, column=1)
        ttk.Label(trn, text="Epochs:")\
            .grid(row=2, column=0, sticky="w")
        self.trn_epochs = ttk.Entry(trn, width=10); self.trn_epochs.insert(0,"30")
        self.trn_epochs.grid(row=2, column=1)
        ttk.Label(trn, text="Batch size:")\
            .grid(row=3, column=0, sticky="w")
        self.trn_batch = ttk.Entry(trn, width=10); self.trn_batch.insert(0,"256")
        self.trn_batch.grid(row=3, column=1)
        ttk.Label(trn, text="Learning rate:")\
            .grid(row=4, column=0, sticky="w")
        self.trn_lr = ttk.Entry(trn, width=10); self.trn_lr.insert(0,"0.001")
        self.trn_lr.grid(row=4, column=1)
        self.trn_start = ttk.Button(trn, text="Start", command=self._start_train)
        self.trn_start.grid(row=5, column=0, pady=10)
        self.trn_stop  = ttk.Button(trn, text="Stop",  command=self._stop_train,
                                    state="disabled")
        self.trn_stop.grid(row=5, column=1)

        # Assist tab
        ast = ttk.Frame(nb); nb.add(ast, text="Assist")
        ttk.Label(ast, text="Select model:")\
            .grid(row=0, column=0, sticky="w")
        self.ast_model = ttk.Combobox(ast, state="readonly", width=18)
        self.ast_model.grid(row=0, column=1)
        self.ast_model.bind("<<ComboboxSelected>>", self._on_model_select)
        ttk.Label(ast, text="Target app:")\
            .grid(row=1, column=0, sticky="w")
        self.ast_app = ttk.Entry(ast, width=20); self.ast_app.grid(row=1, column=1)
        ttk.Label(ast, text="Assist strength α [0–1]:")\
            .grid(row=2, column=0, sticky="w")
        self.ast_alpha = ttk.Entry(ast, width=10); self.ast_alpha.insert(0,"0.3")
        self.ast_alpha.grid(row=2, column=1)
        ttk.Label(ast, text="Toggle key (e.g. F8):")\
            .grid(row=3, column=0, sticky="w")
        self.ast_toggle = ttk.Entry(ast, width=10); self.ast_toggle.insert(0,"F8")
        self.ast_toggle.grid(row=3, column=1)
        self.ast_start = ttk.Button(ast, text="Start", command=self._start_assist)
        self.ast_start.grid(row=4, column=0, pady=10)
        self.ast_stop  = ttk.Button(ast, text="Stop",  command=self._stop_assist,
                                    state="disabled")
        self.ast_stop.grid(row=4, column=1)

        nb.pack(fill="x", padx=10, pady=10)

        # Log area
        self.log = scrolledtext.ScrolledText(self, height=10, state="disabled")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0,5))

        # Status bar
        self.status = ttk.Label(self, text="Status: Idle",
                                relief=tk.SUNKEN, anchor="w")
        self.status.pack(fill="x", side="bottom")

        # Refresh models when switching to Assist tab
        nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _append_log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}"
        # to text widget
        self.log.config(state="normal")
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.config(state="disabled")
        # to file
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def _update_status(self, text):
        self.status.config(text=f"Status: {text}")

    def _on_tab_change(self, event):
        tab = event.widget.tab(event.widget.select(), "text")
        if tab == "Assist":
            self._refresh_models()

    def _refresh_models(self):
        files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
        names = [os.path.splitext(os.path.basename(p))[0] for p in files]
        self.ast_model['values'] = names

    def _on_model_select(self, event):
        name = self.ast_model.get()
        path = os.path.join(MODELS_DIR, f"{name}.pth")
        try:
            ckpt = torch.load(path)
            app = ckpt.get('app_name', '')
            self.ast_app.delete(0, tk.END)
            self.ast_app.insert(0, app)
        except Exception:
            pass

    # --- Record controls ---

    def _start_record(self):
        dur_text = self.rec_dur.get().strip()
        duration = float(dur_text) if dur_text else None
        app_filter = self.rec_filter.get().strip()
        self.record_event.clear()
        self.rec_start.config(state="disabled")
        self.rec_stop.config(state="normal")
        self._update_status("Recording")
        threading.Thread(
            target=record_data,
            kwargs={
                'duration': duration,
                'app_filter': app_filter,
                'log_cb': self._append_log,
                'notify': self.toaster.show_toast,
                'stop_event': self.record_event
            },
            daemon=True
        ).start()

    def _stop_record(self):
        self.record_event.set()
        self.rec_start.config(state="normal")
        self.rec_stop.config(state="disabled")
        self._update_status("Idle")

    # --- Train controls ---

    def _start_train(self):
        name = self.trn_name.get().strip()
        app  = self.trn_app.get().strip()
        if not name:
            messagebox.showerror("Error", "Model name is required")
            return
        self.train_event.clear()
        self.trn_start.config(state="disabled")
        self.trn_stop.config(state="normal")
        self._update_status("Training")
        threading.Thread(
            target=train_model,
            kwargs={
                'epochs': int(self.trn_epochs.get()),
                'batch_size': int(self.trn_batch.get()),
                'lr': float(self.trn_lr.get()),
                'model_name': name,
                'target_app': app,
                'log_cb': self._append_log,
                'notify': self.toaster.show_toast,
                'stop_event': self.train_event
            },
            daemon=True
        ).start()

    def _stop_train(self):
        self.train_event.set()
        self.trn_start.config(state="normal")
        self.trn_stop.config(state="disabled")
        self._update_status("Idle")

    # --- Assist controls ---

    def _start_assist(self):
        model_name = self.ast_model.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model")
            return
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        alpha      = float(self.ast_alpha.get())
        target_app = self.ast_app.get().strip()
        toggle_key = self.ast_toggle.get().strip()
        self.assist_event.clear()
        self.ast_start.config(state="disabled")
        self.ast_stop.config(state="normal")
        self._update_status("Assisting")
        threading.Thread(
            target=run_assist,
            kwargs={
                'model_path': model_path,
                'alpha': alpha,
                'target_app': target_app,
                'toggle_key': toggle_key,
                'log_cb': self._append_log,
                'notify': self.toaster.show_toast,
                'stop_event': self.assist_event
            },
            daemon=True
        ).start()

    def _stop_assist(self):
        self.assist_event.set()
        self.ast_start.config(state="normal")
        self.ast_stop.config(state="disabled")
        self._update_status("Idle")

    def on_closing(self):
        # ensure threads stop
        self.record_event.set()
        self.train_event.set()
        self.assist_event.set()
        self.log_file.close()
        self.destroy()

if __name__ == '__main__':
    app = AimAssistGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
