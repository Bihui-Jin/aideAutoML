import os, random, gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import torch
import threading
import pyglove as pg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import wandb
os.environ['WANDB_API_KEY'] = '51e19a7c4d2d6d577fd64d1d9e64a43fa83ccafa'

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
# Simple MLP model
# ----------------------------
class MLP(nn.Module):
    def __init__(
        self, in_dim, hidden_sizes=(512, 512, 512, 512), dropout=0.05, act="gelu"
    ):
        super().__init__()
        act_fn = nn.GELU() if act == "gelu" else nn.ReLU()
        layers = []
        last = in_dim
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.BatchNorm1d(hs), act_fn]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = hs
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1, 1)


# ----------------------------
# Symbolic Experiment
# ----------------------------
@pg.symbolize
class Experiment:
    def __init__(
        self,
        # --- Model search knobs ---
        classic_arch="mlp@lr=0.000700",
        # pg.oneof(
        #     [
        #         # Encode LR choices into the architecture token to keep signature unchanged.
        #         "mlp@lr=0.000700",
        #         "mlp@lr=0.000705",
        #         "mlp@lr=0.000710",
        #         "mlp@lr=0.000715",
        #         "mlp@lr=0.000720",
        #     ]
        # ),
        # Transformer-specific (fixed and unused here)
        hf_backbone="intfloat/e5-small-v2",
        # pg.oneof(
        #     [
        #         "intfloat/e5-small-v2",
        #     ]
        # ),
    ):
        # Assign
        self.hf_backbone = hf_backbone
        self.classic_arch = classic_arch

        # Fixed knobs
        self.val_size = 0.10
        self.epochs = 7
        self.batch_size = 256
        self.dropout = 0.05
        self.activation = "gelu"
        self.use_amp = True
        self.weight_decay = 0.0
        self.patience = 2

        # Fitted artifacts
        self._train_df = None
        self._test_df = None
        self._y = None
        self._numeric_cols = None
        self._categorical_cols = None
        self._bool_cols = None
        self._spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        self._ct = None
        self.input_dim_ = None

        # DataLoaders
        self.dl_tr = None
        self.dl_va = None
        self.dl_full = None
        self.dl_te = None

        # Model
        self.model_ = None
        self.lr_ = self._parse_lr(self.classic_arch)

        # WandB state
        self._wb_enabled = True
        self._wb_run = None
        self._global_step = 0

    def _parse_lr(self, token):
        # token format: "mlp@lr=0.000710"
        try:
            if "@lr=" in token:
                return float(token.split("@lr=")[1])
        except:
            pass
        return 7.1e-4

    # ------------------------
    # 1) Data Processing (symbolic search space realized)
    # ------------------------
    def data_processing(
        self,
    ):
        tr = pd.read_csv("./input/train.csv")
        te = pd.read_csv("./input/test.csv")
        self._train_df = tr.copy()
        self._test_df = te.copy()
        self._y = tr["Transported"].astype(int).values

        def engineer(df):
            # Cabin split
            cabin = df["Cabin"].fillna("Unknown/0/Unknown").astype(str)
            parts = cabin.str.split("/", expand=True)
            df["CabinDeck"] = parts[0]
            df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
            df["CabinSide"] = parts[2]
            # Group size
            grp = df["PassengerId"].astype(str).str.split("_", expand=True)[0]
            df["GroupSize"] = grp.map(grp.value_counts())
            # Total spend
            for c in self._spend_cols:
                df[c] = df[c].fillna(0.0)
            df["TotalSpend"] = df[self._spend_cols].sum(axis=1)
            # Surname length
            surn = (
                df["Name"]
                .fillna("Unknown Unknown")
                .astype(str)
                .apply(lambda x: x.split()[-1] if len(x.split()) > 0 else "Unknown")
            )
            df["SurnameLen"] = surn.apply(len)
            return df

        self._train_df = engineer(self._train_df)
        self._test_df = engineer(self._test_df)

        self._numeric_cols = [
            "Age",
            "CabinNum",
            "GroupSize",
            "TotalSpend",
            "SurnameLen",
        ] + self._spend_cols
        self._numeric_cols = [
            c for c in self._numeric_cols if c in self._train_df.columns
        ]

        self._bool_cols = [
            c for c in ["CryoSleep", "VIP"] if c in self._train_df.columns
        ]

        self._categorical_cols = [
            c
            for c in ["HomePlanet", "Destination", "CabinDeck", "CabinSide"]
            if c in self._train_df.columns
        ]
        return

    # ------------------------
    # 2) Model (symbolic architecture)
    # ------------------------
    def model(
        self,
    ):
        return MLP(
            self.input_dim_,
            hidden_sizes=(512, 512, 512, 512),
            dropout=self.dropout,
            act=self.activation,
        ).to(DEVICE)

    # ------------------------
    # 3) Optimizer (symbolic strategy)
    # ------------------------
    def _optimizer(
        self,
    ):
        return torch.optim.Adam(
            self.model_.parameters(), lr=float(self.lr_), weight_decay=self.weight_decay
        )

    # ------------------------
    # 4) Training (symbolic strategy)
    # ------------------------
    def training(
        self,
    ):
        optimizer = self._optimizer()
        scaler = torch.cuda.amp.GradScaler(
            enabled=(self.use_amp and torch.cuda.is_available())
        )
        criterion = nn.BCEWithLogitsLoss()

        # wandb log gradients automatically (optional)
        if self._wb_run is not None:
            wandb.watch(self.model_, log="gradients", log_freq=100)

        best_score = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(int(self.epochs)):
            self.model_.train()
            optimizer.zero_grad(set_to_none=True)
            for Xb, yb in self.dl_tr:
                Xb = Xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).view(-1, 1)
                with torch.cuda.amp.autocast(
                    enabled=(self.use_amp and torch.cuda.is_available())
                ):
                    logits = self.model_(Xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()

                # wandb grad norm
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = self._grad_l2_norm(self.model_)
                lr_now = optimizer.param_groups[0]["lr"]

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # wandb log per batch
                # log per batch
                self._global_step += 1
                self._wandb_log(
                    {
                        "train/loss": float(loss.detach().cpu().item()),
                        "train/grad_norm": grad_norm,
                        "train/lr": lr_now,
                        "train/epoch": epoch,
                    }
                )

            # Validate
            self.model_.eval()
            preds_all, tgts_all = [], []
            val_losses = []
            with torch.no_grad():
                for Xv, yv in self.dl_va:
                    Xv = Xv.to(DEVICE, non_blocking=True)
                    yv = yv.to(DEVICE, non_blocking=True).view(-1, 1)
                    logits = self.model_(Xv)

                    # Compute validation loss
                    val_loss = criterion(logits, yv)
                    val_losses.append(float(val_loss.detach().cpu().item()))
                    
                    probs = torch.sigmoid(logits).view(-1)
                    preds = (probs >= 0.5).long()
                    preds_all.append(preds.detach().cpu().numpy())
                    tgts_all.append(yv.view(-1).long().detach().cpu().numpy())
            if len(preds_all) > 0:
                y_pred = np.concatenate(preds_all)
                y_true = np.concatenate(tgts_all)
                score = (y_pred == y_true).mean()
            else:
                score = 0.0

            # wandb compute average validation loss
            avg_val_loss = np.mean(val_losses) if val_losses else 0.0

            # wandb log val accuracy (learning curve)
            self._wandb_log({"val/accuracy": float(score), 
                             "val/loss": float(avg_val_loss),
                             "train/epoch": epoch})


            if score > best_score:
                best_score = score
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return best_score

    # ------------------------
    # 5) Evaluation
    # ------------------------
    def evaluation(self, y_true, y_prob):
        preds = (y_prob >= 0.5).astype(int)
        return (preds == y_true).mean()

    # ------------------------
    # 6) Run end-to-end: builds features, trains, evaluates, and writes submission.
    # ------------------------
    def run(self):
        # Do not add print/logging to avoid overhead during NAS
        set_seed(42)
        self.data_processing()
        # wandb init
        self._wandb_init()

        used_numeric = [c for c in self._numeric_cols]
        used_categorical = [c for c in self._categorical_cols] + [
            c for c in self._bool_cols
        ]
        X_df = self._train_df[used_numeric + used_categorical]
        Xt_df = self._test_df[used_numeric + used_categorical]
        y = self._y

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler(with_mean=True)),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
            ]
        )
        self._ct = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, used_numeric),
                ("cat", categorical_transformer, used_categorical),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        X_tr_df, X_va_df, y_tr, y_va = train_test_split(
            X_df, y, test_size=self.val_size, stratify=y, random_state=42
        )
        self._ct.fit(X_tr_df)
        X_tr = self._ct.transform(X_tr_df)
        X_va = self._ct.transform(X_va_df)
        X_te = self._ct.transform(Xt_df)

        if hasattr(X_tr, "toarray"):
            X_tr = X_tr.toarray()
            X_va = X_va.toarray()
            X_te = X_te.toarray()
        X_tr = X_tr.astype(np.float32)
        X_va = X_va.astype(np.float32)
        X_te = X_te.astype(np.float32)

        self.input_dim_ = X_tr.shape[1]

        bs = int(self.batch_size)
        ds_tr = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr.astype(np.float32)),
        )
        ds_va = TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(y_va.astype(np.float32)),
        )
        self.dl_tr = DataLoader(
            ds_tr, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True
        )
        self.dl_va = DataLoader(
            ds_va,
            batch_size=max(1, bs * 2),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Build & train
        self.model_ = self.model()
        best_val = self.training()

        # Validation predictions
        self.model_.eval()
        va_probs = []
        with torch.no_grad():
            for Xb, _ in self.dl_va:
                Xb = (
                    Xb[0].to(DEVICE) if isinstance(Xb, (list, tuple)) else Xb.to(DEVICE)
                )
                logits = self.model_(Xb)
                probs = torch.sigmoid(logits).view(-1)
                va_probs.append(probs.detach().cpu().numpy())
        va_probs = (
            np.concatenate(va_probs)
            if len(va_probs) > 0
            else np.zeros_like(y_va, dtype=float)
        )
        score = self.evaluation(y_va, va_probs)

        # Train on all data for final test predictions (a bit shorter to limit drift)
        X_full = self._ct.transform(X_df)
        if hasattr(X_full, "toarray"):
            X_full = X_full.toarray()
        X_full = X_full.astype(np.float32)
        ds_full = TensorDataset(
            torch.tensor(X_full, dtype=torch.float32),
            torch.tensor(y.astype(np.float32)),
        )
        self.dl_full = DataLoader(
            ds_full, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True
        )
        ds_te = TensorDataset(torch.tensor(X_te, dtype=torch.float32))
        self.dl_te = DataLoader(
            ds_te,
            batch_size=max(1, bs * 2),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        self.model_ = self.model()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=float(self.lr_), weight_decay=self.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=(self.use_amp and torch.cuda.is_available())
        )
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(max(1, int(self.epochs) - 1)):
            self.model_.train()
            optimizer.zero_grad(set_to_none=True)
            for Xb, yb in self.dl_full:
                Xb = Xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).view(-1, 1)
                with torch.cuda.amp.autocast(
                    enabled=(self.use_amp and torch.cuda.is_available())
                ):
                    logits = self.model_(Xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()

                # wandb grad norm
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = self._grad_l2_norm(self.model_)
                lr_now = optimizer.param_groups[0]["lr"]

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # wandb log per epoch
                self._global_step += 1
                self._wandb_log(
                    {
                        "full/loss": float(loss.detach().cpu().item()),
                        "full/grad_norm": grad_norm,
                        "full/lr": lr_now,
                        "full/epoch": epoch,
                    }
                )

        self.model_.eval()
        test_probs = []
        with torch.no_grad():
            for (Xb,) in self.dl_te:
                Xb = Xb.to(DEVICE, non_blocking=True)
                logits = self.model_(Xb)
                probs = torch.sigmoid(logits).view(-1)
                test_probs.append(probs.detach().cpu().numpy())
        test_probs = (
            np.concatenate(test_probs)
            if len(test_probs) > 0
            else np.zeros(self._test_df.shape[0], dtype=float)
        )

        # Finish WandB
        if self._wb_run is not None:
            wandb.finish()

        return (score, test_probs)
    

    # ------------------------
    # WandB helpers
    # ------------------------
    def _wandb_init(self):
        if self._wb_enabled and self._wb_run is None:
            self._wb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "mle-bench-nas"),
                name=os.getenv("WANDB_RUN_NAME", f"exp-{os.getpid()}"),
                config={
                    "arch": self.classic_arch,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "dropout": self.dropout,
                    "activation": self.activation,
                    "weight_decay": self.weight_decay,
                    "use_amp": self.use_amp,
                    "val_size": self.val_size,
                    "lr": float(self.lr_),
                },
                # set WANDB_MODE=offline if needed
            )
    
    def _wandb_log(self, data: dict):
        if self._wb_run is not None:
            wandb.log(data, step=self._global_step)

    @staticmethod
    def _grad_l2_norm(model: nn.Module) -> float:
        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                sq += float(g.norm(2).item() ** 2)
        return sq ** 0.5


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
    test_ids = pd.read_csv("./input/test.csv")["PassengerId"]
    preds = np.asarray(best_test_probs)
    n = min(len(test_ids), len(preds))
    test_ids = test_ids.iloc[:n].reset_index(drop=True)
    preds = preds[:n]
    submission = pd.DataFrame({"PassengerId": test_ids, "Transported": (preds >= 0.5)})
    submission.to_csv("submission/submission.csv", index=False)
# Do not add new prints/logging.
