import os, random, gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import torch
import threading
import pyglove as pg

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

import wandb
os.environ['WANDB_API_KEY'] = '51e19a7c4d2d6d577fd64d1d9e64a43fa83ccafa'
# Disable WandB console output to avoid TTY issues
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['WANDB_SILENT'] = 'true'

# ----------------------------
# Repro, Device, Utilities (kept from the template)
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
set_seed()


# ----------------------------
# Outer NAS loop with timeout (kept from the template)
# ----------------------------
def run_with_timeout(func, timeout_sec):
    """Run function with timeout, return (success, result)"""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_sec)

    if thread.is_alive():
        # Thread is still running, timeout occurred
        return False, None
    elif exception[0]:
        raise exception[0]
    else:
        return True, result[0]


# ----------------------------
# Simple dataset utilities
# ----------------------------
class ClassicTensorDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx], None
        return self.X[idx], self.y[idx]


def collate_classic(batch):
    xs, ys = [], []
    for x, y in batch:
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    if ys[0] is None:
        return X, None
    Y = torch.stack(ys)
    return X, Y


# ----------------------------
# Tabular architectures
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, activation="mish", batch_norm=False):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "mish": nn.Mish(),
        }
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.act = acts.get(activation, nn.Mish())
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.bn2(h)
        return self.act(h + x)


class ResMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=256,
        num_blocks=2,
        dropout=0.3,
        activation="mish",
        batch_norm=False,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


# ----------------------------
# Symbolic Experiment
# ----------------------------
@pg.symbolize
class Experiment:
    def __init__(
        self,
        # --- Data processing search knobs ---
        # --- Model search knobs ---
        classic_arch="tabresnet",
        # Transformer-specific
        hf_backbone="BAAI/bge-base-en-v1.5",
        # --- Optimizer/search knobs (finalized/locked where appropriate) ---
        opt_name="AdamW",
        lr_classic=1.0e-3,
        dropout=0.26,
        hidden_dim=224,
        num_blocks= 3,
        batch_norm=False,
        weight_decay=1e-4,
        activation="mish",
        # --- Training/eval knobs ---
        epochs=8,
        batch_size_classic=256,
        gradient_accumulation_steps=2,
        use_amp=True,
        early_stop=True,
        patience=3,
        valid_size=0.06,
    ):
        # Assign
        self.hf_backbone = hf_backbone
        self.classic_arch = classic_arch

        self.opt_name = opt_name
        self.lr_classic = lr_classic
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        self.weight_decay = weight_decay
        self.activation = activation

        self.epochs = epochs
        self.batch_size_classic = batch_size_classic
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.early_stop = early_stop
        self.patience = patience
        self.valid_size = valid_size

        # Fitted artifacts
        self.input_dim = 0
        self.num_labels = 2
        self.train_loader = None
        self.valid_loader = None
        self.full_loader = None
        self.test_loader = None
        self.y_valid_np = None
        self.test_ids = None

        # Store preprocessors for refit
        self._imputer_full = None
        self._scaler_full = None
        self._X_full = None
        self._y_full = None
        self._X_test_full = None

        # WandB
        self._wb_enabled = True
        self._wb_run = None
        self._global_step = 0
        self._full_step = 0


    def _wandb_init(self):
        if self._wb_enabled and self._wb_run is None:
            self._wb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "playground-series-s3e18"),
                name=os.getenv("WANDB_RUN_NAME", f"nas-ResMLP-{os.getpid()}"),
                config={
                    "model": "ResMLP",
                    "opt_name": self.opt_name,
                    "learning_rate": float(self.lr_classic),
                    "dropout": float(self.dropout),
                    "hidden_dim": int(self.hidden_dim),
                    "num_blocks": int(self.num_blocks),
                    "batch_norm": bool(self.batch_norm),
                    "weight_decay": float(self.weight_decay),
                    "activation": self.activation,
                    "epochs": int(self.epochs),
                    "batch_size_classic": int(self.batch_size_classic),
                    "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
                    "use_amp": bool(self.use_amp),
                    "early_stop": bool(self.early_stop),
                    "patience": int(self.patience),
                    "valid_size": float(self.valid_size),
                },
                settings=wandb.Settings(console="off", silent=True),
            )
            # Axes mapping
            wandb.define_metric("iteration")
            wandb.define_metric("train/*", step_metric="iteration")
            wandb.define_metric("epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("full/iteration")
            wandb.define_metric("full/*", step_metric="full/iteration")

    def _wandb_log(self, data: dict, *, step: int | None = None):
        if self._wb_run is not None:
            if step is None:
                wandb.log(data)
            else:
                wandb.log(data, step=step)

    @staticmethod
    def _grad_l2_norm(model: nn.Module) -> float:
        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                sq += float(g.norm(2).item() ** 2)
        return sq ** 0.5
    # ------------------------
    # 1) Data Processing (symbolic search space realized)
    # ------------------------
    def data_processing(
        self,
    ):
        # Build raw fields
        train_df = pd.read_csv("./input/train.csv")
        test_df = pd.read_csv("./input/test.csv")
        target_cols = ["EC1", "EC2"]
        all_targets = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]
        feat_cols = [c for c in train_df.columns if c not in ["id"] + all_targets]

        # Stratified split on joint EC1/EC2
        stratify_key = train_df["EC1"].astype(str) + "_" + train_df["EC2"].astype(str)
        trn_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=float(self.valid_size),
            random_state=42,
            stratify=stratify_key,
        )
        trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Impute + scale
        imputer = SimpleImputer(strategy="median")
        X_trn = imputer.fit_transform(trn_df[feat_cols])
        X_val = imputer.transform(val_df[feat_cols])
        scaler = RobustScaler()
        X_trn = scaler.fit_transform(X_trn)
        X_val = scaler.transform(X_val)
        X_tst = scaler.transform(imputer.transform(test_df[feat_cols]))

        y_trn = trn_df[target_cols].values.astype(np.float32)
        y_val = val_df[target_cols].values.astype(np.float32)

        # Refit preproc on full train for final model
        imputer_full = SimpleImputer(strategy="median")
        X_full = imputer_full.fit_transform(train_df[feat_cols])
        scaler_full = RobustScaler()
        X_full = scaler_full.fit_transform(X_full)
        X_test_full = scaler_full.transform(imputer_full.transform(test_df[feat_cols]))
        y_full = train_df[target_cols].values.astype(np.float32)

        self._imputer_full = imputer_full
        self._scaler_full = scaler_full
        self._X_full = X_full.astype(np.float32)
        self._y_full = y_full
        self._X_test_full = X_test_full.astype(np.float32)

        trn_ds = ClassicTensorDataset(X_trn.astype(np.float32), y_trn)
        val_ds = ClassicTensorDataset(X_val.astype(np.float32), y_val)
        full_ds = ClassicTensorDataset(self._X_full, self._y_full)
        tst_ds = ClassicTensorDataset(X_tst.astype(np.float32), None)

        self.train_loader = DataLoader(
            trn_ds,
            batch_size=int(self.batch_size_classic),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_classic,
        )
        self.valid_loader = DataLoader(
            val_ds,
            batch_size=int(self.batch_size_classic),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_classic,
        )
        self.full_loader = DataLoader(
            full_ds,
            batch_size=int(self.batch_size_classic),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_classic,
        )
        self.test_loader = DataLoader(
            tst_ds,
            batch_size=int(self.batch_size_classic),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_classic,
        )

        self.y_valid_np = y_val
        self.input_dim = X_trn.shape[1]
        self.test_ids = test_df["id"].values
        return

    # ------------------------
    # 2) Model (symbolic architecture)
    # ------------------------
    def model(
        self,
    ):
        # Model archs in PyGlove architecture space that can achieve best performance to such task type
        if self.classic_arch == "tabresnet":
            return ResMLP(
                in_dim=self.input_dim,
                out_dim=self.num_labels,
                hidden_dim=int(self.hidden_dim),
                num_blocks=int(self.num_blocks),
                dropout=float(self.dropout),
                activation=self.activation,
                batch_norm=bool(self.batch_norm),
            ).to(DEVICE)
        # Fallback
        return ResMLP(
            in_dim=self.input_dim,
            out_dim=self.num_labels,
            hidden_dim=int(self.hidden_dim),
            num_blocks=int(self.num_blocks),
            dropout=float(self.dropout),
            activation=self.activation,
            batch_norm=bool(self.batch_norm),
        ).to(DEVICE)

    # ------------------------
    # 3) Optimizer (symbolic strategy)
    # ------------------------
    def _optimizer(self, model):
        if self.opt_name == "AdamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=float(self.lr_classic),
                weight_decay=float(self.weight_decay),
            )
        else:
            return torch.optim.AdamW(
                model.parameters(),
                lr=float(self.lr_classic),
                weight_decay=float(self.weight_decay),
            )

    # ------------------------
    # 4) Training (symbolic strategy)
    # ------------------------
    def training(
        self,
    ):
        model = self.model()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = self._optimizer(model)
        scaler = torch.cuda.amp.GradScaler(
            enabled=(bool(self.use_amp) and DEVICE.type == "cuda")
        )

        best_score = -1e9
        best_state = None
        no_improve = 0
        global_iter = 0

        for epoch in range(int(self.epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for step, (feats, targets) in enumerate(self.train_loader):
                feats = feats.to(DEVICE)
                targets = targets.to(DEVICE)
                with torch.cuda.amp.autocast(
                    enabled=(bool(self.use_amp) and DEVICE.type == "cuda")
                ):
                    logits = model(feats)
                    loss = criterion(logits, targets)
                loss = loss / int(self.gradient_accumulation_steps)
                scaler.scale(loss).backward()

                # Compute unscaled grad norm at accumulation boundary
                at_boundary = (step + 1) % int(self.gradient_accumulation_steps) == 0 or (
                    step + 1
                ) == len(self.train_loader)

                # wandb logging
                if at_boundary and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = self._grad_l2_norm(model) if at_boundary else float("nan")
                lr_now = optimizer.param_groups[0]["lr"]
                global_iter += 1
                self._global_step = global_iter
                # Log per batch
                self._wandb_log(
                    {
                        "iteration": global_iter,
                        "train/loss": float(loss.detach().cpu().item()),
                        "train/grad_norm": float(grad_norm) if grad_norm == grad_norm else grad_norm,
                        "train/gradiant_norm": float(grad_norm) if grad_norm == grad_norm else grad_norm,  # alias
                        "train/lr": float(lr_now),
                    },
                    step=global_iter,
                )

                if at_boundary:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            # Validation
            model.eval()
            probs, tgts = [], []
            with torch.no_grad():
                for feats, targets in self.valid_loader:
                    feats = feats.to(DEVICE)
                    logits = model(feats)
                    p = torch.sigmoid(logits).cpu().numpy()
                    probs.append(p)
                    tgts.append(targets.numpy())
            probs = np.concatenate(probs, axis=0)
            tgts = np.concatenate(tgts, axis=0)
            try:
                auc1 = roc_auc_score(tgts[:, 0], probs[:, 0])
                auc2 = roc_auc_score(tgts[:, 1], probs[:, 1])
                score = 0.5 * (auc1 + auc2)
            except Exception:
                score = 0.5

            self._wandb_log({"epoch": epoch, "val/auc1": auc1, "val/auc2": auc2, "val/accuracy": score})

            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if bool(self.early_stop) and no_improve >= int(self.patience):
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_score

    # ------------------------
    # 5) Evaluation
    # ------------------------
    def evaluation(self, model):
        # Do not add print/logging here to avoid overhead during NAS
        # Implementing the evaluation metric for such task type from the competition description
        model.eval()
        probs, tgts = [], []
        with torch.no_grad():
            for feats, targets in self.valid_loader:
                feats = feats.to(DEVICE)
                logits = model(feats)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
                tgts.append(targets.numpy())
        probs = np.concatenate(probs, axis=0)
        tgts = np.concatenate(tgts, axis=0)
        try:
            auc1 = roc_auc_score(tgts[:, 0], probs[:, 0])
            auc2 = roc_auc_score(tgts[:, 1], probs[:, 1])
            score = 0.5 * (auc1 + auc2)
        except Exception:
            score = 0.5
        return score

    # ------------------------
    # 6) Run end-to-end: builds features, trains, evaluates, and writes submission.
    # ------------------------
    def run(self):
        # Do not add print/logging to avoid overhead during NAS
        # Build dense features from the symbolic data-processing pipeline
        self.data_processing()

        # Init WandB
        self._wandb_init()
        self._full_step = 0

        # Hold-out validation for fast feedback
        model, _ = self.training()

        # Validation
        score = self.evaluation(model)

        # Train on all data for final test predictions
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = self._optimizer(model)
        scaler = torch.cuda.amp.GradScaler(
            enabled=(bool(self.use_amp) and DEVICE.type == "cuda")
        )
        max_epochs = max(1, int(self.epochs) - 1)
        for epoch in range(max_epochs):
            optimizer.zero_grad(set_to_none=True)
            for step, (feats, targets) in enumerate(self.full_loader):
                feats = feats.to(DEVICE)
                targets = targets.to(DEVICE)
                with torch.cuda.amp.autocast(
                    enabled=(bool(self.use_amp) and DEVICE.type == "cuda")
                ):
                    logits = model(feats)
                    loss = criterion(logits, targets)
                loss = loss / int(self.gradient_accumulation_steps)
                scaler.scale(loss).backward()
                
                at_boundary = (step + 1) % int(self.gradient_accumulation_steps) == 0 or (
                    step + 1
                ) == len(self.full_loader)

                # wandb logging
                if at_boundary and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = self._grad_l2_norm(model) if at_boundary else float("nan")
                self._full_step += 1
                self._wandb_log(
                    {
                        "full/iteration": self._full_step,
                        "full/loss": float(loss.detach().cpu().item()),
                        "full/grad_norm": float(grad_norm) if grad_norm == grad_norm else grad_norm,
                        "full/gradiant_norm": float(grad_norm) if grad_norm == grad_norm else grad_norm,  # alias
                        "full/lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )

                if at_boundary:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        # Predict test
        model.eval()
        test_probs = []
        with torch.no_grad():
            for feats, _ in self.test_loader:
                feats = feats.to(DEVICE)
                logits = model(feats)
                p = torch.sigmoid(logits).cpu().numpy()
                test_probs.append(p)
        test_probs = np.vstack(test_probs)

        # Finish WandB
        if self._wb_run is not None:
            wandb.finish()

        # Return both validation score and test predictions
        return score, test_probs


exp_template = Experiment()


def smoke_test():
    pass


# ----------------------------
# Main Execution Code Chunk (kept from the origin)
# Instantiate a compact search space (kept from the template)
# Do NOT modify below this line until the "Finished" line
# ----------------------------
exp_template = Experiment()

run_smoke_test = False
best_score, best_exp = None, None
best_test_probs = None
# authentication key for models from Huggingface
auth_token = os.getenv("HUGGINGFACE_KEY")
_timeout = 405
run, trial = 1, 1

result = run_with_timeout(exp_template.run, timeout_sec=_timeout)
success, (best_score, best_test_probs) = result
print(f"Best Validation Score: {best_score:.6f}")

# def full_search():
#     global run, trial, best_score, best_exp, best_test_probs, _timeout
#     algo = pg.evolution.regularized_evolution(
#         population_size=64, tournament_size=6, seed=42
#     )

#     with open("/home/agent/output.txt", "w") as output_file:
#         output_file.write("Model performance\n")
#     # Limit the upper bound of total trial to avoid infinite loop
#     for i, (exp, feedback) in enumerate(
#         pg.sample(exp_template, algo, num_examples=164)
#     ):
#         # Limit to 60 trial
#         if trial > 60:
#             break

#         with open("/home/agent/running.txt", "w") as running_exp:
#             running_exp.write(f"{exp}")

#         result = run_with_timeout(exp.run, timeout_sec=_timeout)

#         if not result[0]:
#             # Give it a bad score
#             feedback(float("-inf"))
#             # Clean up GPU memory
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             gc.collect()
#             continue

#         success, (score, test_probs) = result
#         feedback(score)

#         with open("/home/agent/output.txt", "a") as output_file:
#             output_file.write(f"\n=== Trial {run} ===\n")
#             output_file.write(f"Validation score: {score:.6f}\n")
#             output_file.write(f"Tested parameters: {exp}\n")

#         # Track best
#         if best_score is None or score > best_score:
#             best_score = score
#             best_test_probs = test_probs
#             best_exp = exp

#         score = float("-inf")
#         run += 1
#         trial += 1 if i > 64 else 0  # Warm-up phase does not count towards trial count

#     print(f"\n=== Search Complete ===")
#     print(f"Best Validation Score: {best_score:.6f}")
#     print(f"Best Parameters: {best_exp}")


# if run_smoke_test:
#     smoke_test()
# run_smoke_test = False
# full_search()
# ----------------------------
# Finished
# Do not modify above this line
# ----------------------------

# ----------------------------
# Ensure best submission is saved
# Fill submission in according to the competition's submission format
# ----------------------------
os.makedirs("./submission", exist_ok=True)
if best_test_probs is not None:
    submission = pd.DataFrame(
        {
            "id": pd.read_csv("./input/test.csv")["id"].values[
                : best_test_probs.shape[0]
            ],
            "EC1": best_test_probs[:, 0],
            "EC2": best_test_probs[:, 1],
        }
    )
    submission.to_csv("submission/submission.csv", index=False)
# Do not add new prints/logging.
