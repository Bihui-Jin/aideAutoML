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

from sklearn.model_selection import StratifiedKFold
import wandb
from pathlib import Path
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
os.environ["WANDB_API_KEY"] = "51e19a7c4d2d6d577fd64d1d9e64a43fa83ccafa"
# Disable WandB console output to avoid TTY issues
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"


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
        # If batch_size == 1 and we are in training mode,
        # skip BatchNorm1d to avoid "Expected more than 1 value per channel" error.
        if self.training and x.size(0) == 1:
            out = x
            for layer in self.net:
                if isinstance(layer, nn.BatchNorm1d):
                    # Skip BN when batch size is 1 during training
                    continue
                out = layer(out)
            return out.view(-1, 1)
        else:
            return self.net(x).view(-1, 1)


# ----------------------------
# Symbolic Experiment
# ----------------------------
@pg.symbolize
class Experiment:
    def __init__(
        self,
        # --- Model search knobs ---
        classic_arch= pg.oneof(
            [
                # Encode LR choices into the architecture token to keep signature unchanged.
                "mlp@lr=0.000600",
                "mlp@lr=0.000625",
                "mlp@lr=0.000650",
                "mlp@lr=0.000675",
                "mlp@lr=0.000700",
            ]
        ),
        # Transformer-specific (fixed and unused here)
        hf_backbone="intfloat/e5-small-v2",
        # pg.oneof(
        #     [
        #         "intfloat/e5-small-v2",
        #     ]
        # ),
        val_size = pg.oneof([0.05, 0.1]),
        batch_size = pg.oneof([1, 64, 75, 256]),
        dropout = pg.oneof([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]),
        weight_decay = pg.oneof([0, 2e-4, 1e-4, 5e-5, 1e-5]),
        hidden_sizes = pg.oneof([(1024, 512, 256), (512, 512, 512, 512), (512, 512, 256, 128), (512, 512, 512), (1024, 512, 256, 128)]),

    ):
        # Assign
        self.hf_backbone = hf_backbone
        self.classic_arch = classic_arch

        # Fixed knobs
        self.val_size = val_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.hidden_sizes = hidden_sizes
        self.epochs = 30
        self.activation = "gelu"
        self.use_amp = True
        self.patience = 3

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
        self._wb_enabled = False
        self._wb_run = None
        self._global_step = 0

        self.n_folds = 5  # Add k-fold parameter
        self.use_kfold_full = True  # Whether to use k-fold for full training
        self.checkpoint_dir = "./checkpoints"
        self.save_best_only = True
        self.checkpoint_every_n_epochs = 1  # Save checkpoint every N epochs
        self.scheduler_step_size = 1
        self.scheduler_gamma = 0.95

    def _parse_lr(self, token):
        # token format: "mlp@lr=0.000710"
        try:
            if "@lr=" in token:
                return float(token.split("@lr=")[1])
        except:
            pass
        return 7.1e-4

    def _save_checkpoint(self, epoch, score, is_best=False):
        """Save model checkpoint"""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_.state_dict(),
            'score': score,
            'arch': self.classic_arch,
            'lr': float(self.lr_),
            'input_dim': self.input_dim_,
        }
        
        if is_best:
            # Save best model
            best_path = Path(self.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            # self._wandb_log({"checkpoint/best_score": score, "checkpoint/best_epoch": epoch})
            
            # Also save metadata
            metadata = {
                'epoch': epoch,
                'score': float(score),
                'arch': self.classic_arch,
                'lr': float(self.lr_),
            }
            with open(Path(self.checkpoint_dir) / "best_model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        if not self.save_best_only:
            # Save periodic checkpoint
            if epoch % self.checkpoint_every_n_epochs == 0:
                epoch_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, epoch_path)

    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not Path(checkpoint_path).exists():
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model_.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

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
            hidden_sizes=self.hidden_sizes,
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
        # scheduler = CosineAnnealingLR(
        #     optimizer, T_max=int(self.epochs), eta_min=float(self.lr_) * 0.1
        # )
        scaler = torch.cuda.amp.GradScaler(
            enabled=(self.use_amp and torch.cuda.is_available())
        )
        criterion = nn.BCEWithLogitsLoss()

        # wandb log gradients automatically (optional)
        # if self._wb_run is not None:
        #     wandb.watch(self.model_, log="gradients", log_freq=100)

        best_score = -1.0
        best_state = None
        no_improve = 0
        best_epoch = -1

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

                # # wandb grad norm
                # if scaler.is_enabled():
                #     scaler.unscale_(optimizer)
                grad_norm = self._grad_l2_norm(self.model_)
                lr_now = optimizer.param_groups[0]["lr"]

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # # wandb log per batch
                # # log per batch
                # self._global_step += 1
                # self._wandb_log(
                #     {
                #         "train/loss": float(loss.detach().cpu().item()),
                #         "train/grad_norm": grad_norm,
                #         "train/lr": lr_now,
                #         "train/epoch": epoch,
                #     }
                # )

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

            # # wandb log val accuracy (learning curve)
            # self._wandb_log(
            #     {
            #         "val/accuracy": float(score),
            #         "val/loss": float(avg_val_loss),
            #         "train/epoch": epoch,
            #     }
            # )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
                # Save best checkpoint
                self._save_checkpoint(epoch, score, is_best=True)
                no_improve = 0
            else:
                no_improve += 1
                if not self.save_best_only:
                    self._save_checkpoint(epoch, score, is_best=False)
                if no_improve > self.patience:
                    break
        
            # # Step scheduler
            # scheduler.step()
            # # Log updated LR after scheduler step
            # updated_lr = optimizer.param_groups[0]["lr"]
            # self._wandb_log({
            #     "train/lr_after_scheduler": updated_lr,
            #     "train/epoch": epoch,
            # })

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            print(f"Loaded best model from epoch {best_epoch} with score {best_score:.6f}")
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
        # self._wandb_init()

        used_numeric = [c for c in self._numeric_cols]
        used_categorical = [c for c in self._categorical_cols] + [c for c in self._bool_cols]
        X_df = self._train_df[used_numeric + used_categorical]
        Xt_df = self._test_df[used_numeric + used_categorical]
        y = self._y

        # 5-fold ensembling path
        if getattr(self, "n_folds", 1) and int(self.n_folds) > 1:
            skf = StratifiedKFold(n_splits=int(self.n_folds), shuffle=True, random_state=42)
            fold_scores: list[float] = []
            fold_test_probs: list[np.ndarray] = []
            bs = int(self.batch_size)
            old_ckpt_dir = self.checkpoint_dir

            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_df, y)):
                X_tr_df = X_df.iloc[tr_idx]
                X_va_df = X_df.iloc[va_idx]
                y_tr = y[tr_idx]
                y_va = y[va_idx]

                # Build a fresh preprocessor per fold
                numeric_transformer = Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="mean"))
                    # ,
                        #    ("scaler", StandardScaler(with_mean=True))
                           ]
                )
                categorical_transformer = Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True))]
                )
                self._ct = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, used_numeric),
                        ("cat", categorical_transformer, used_categorical),
                    ],
                    remainder="drop",
                    sparse_threshold=0.3,
                )

                self._ct.fit(X_tr_df)
                X_tr = self._ct.transform(X_tr_df)
                X_va = self._ct.transform(X_va_df)
                X_te = self._ct.transform(Xt_df)

                if hasattr(X_tr, "toarray"):
                    X_tr = X_tr.toarray(); X_va = X_va.toarray(); X_te = X_te.toarray()
                X_tr = X_tr.astype(np.float32); X_va = X_va.astype(np.float32); X_te = X_te.astype(np.float32)

                self.input_dim_ = X_tr.shape[1]

                ds_tr = TensorDataset(
                    torch.tensor(X_tr, dtype=torch.float32),
                    torch.tensor(y_tr.astype(np.float32)),
                )
                ds_va = TensorDataset(
                    torch.tensor(X_va, dtype=torch.float32),
                    torch.tensor(y_va.astype(np.float32)),
                )
                self.dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
                self.dl_va = DataLoader(ds_va, batch_size=max(1, bs * 2), shuffle=False, num_workers=0, pin_memory=True)

                # Model per fold
                self.model_ = self.model()

                # Isolate checkpoints per fold
                self.checkpoint_dir = str(Path(old_ckpt_dir) / f"fold_{fold_idx}")

                # Train and track best fold score
                best_val = self.training()
                fold_scores.append(float(best_val))

                # Inference on test for this fold
                ds_te = TensorDataset(torch.tensor(X_te, dtype=torch.float32))
                dl_te = DataLoader(ds_te, batch_size=max(1, bs * 2), shuffle=False, num_workers=0, pin_memory=True)
                test_probs = []
                self.model_.eval()
                with torch.no_grad():
                    for (Xb,) in dl_te:
                        Xb = Xb.to(DEVICE, non_blocking=True)
                        logits = self.model_(Xb)
                        probs = torch.sigmoid(logits).view(-1)
                        test_probs.append(probs.detach().cpu().numpy())
                test_probs = (
                    np.concatenate(test_probs)
                    if len(test_probs) > 0
                    else np.zeros(self._test_df.shape[0], dtype=float)
                )
                fold_test_probs.append(test_probs)

                # restore
                self.checkpoint_dir = old_ckpt_dir

                # # optional: log per-fold
                # self._wandb_log({"cv/fold": fold_idx, "cv/fold_val_acc": float(best_val)})

            # Ensemble by averaging fold probabilities
            best_val = float(np.mean(fold_scores))
            best_test_probs = np.mean(np.vstack(fold_test_probs), axis=0)

            # self._wandb_log({
            #     "cv/val_mean": best_val,
            #     "cv/val_std": float(np.std(fold_scores)),
            #     "cv/n_folds": int(self.n_folds),
            # })

            # if self._wb_run is not None:
            #     wandb.finish()

            return (best_val, best_test_probs)

    # ------------------------
    # WandB helpers
    # ------------------------
    def _wandb_init(self):
        if self._wb_enabled and self._wb_run is None:
            self._wb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "spaceship-titanic"),
                name=os.getenv("WANDB_RUN_NAME", f"nas-mlp-{os.getpid()}"),
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
                settings=wandb.Settings(console="off", silent=True),
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
        return sq**0.5


# exp_template = Experiment()


# def smoke_test():
#     pass


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
_timeout = 600
run, trial = 1, 1

# global run, trial, best_score, best_exp, best_test_probs, _timeout
# algo = pg.evolution.regularized_evolution(
#     population_size=64, tournament_size=6, seed=42
# )

# with open("/home/agent/output.txt", "w") as output_file:
#     output_file.write("Model performance\n")
# Limit the upper bound of total trial to avoid infinite loop
for exp, feedback in pg.sample(exp_template, pg.geno.Sweeping()):
    # with open("/home/agent/running.txt", "w") as running_exp:
    #     running_exp.write(f"{exp}")

    result = run_with_timeout(exp.run, timeout_sec=_timeout)

    if not result[0]:
        # Give it a bad score
        feedback(float("-inf"))
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        continue

    success, (score, test_probs) = result
    feedback(score)

    # with open("/home/agent/output.txt", "a") as output_file:
    #     output_file.write(f"\n=== Trial {run} ===\n")
    #     output_file.write(f"Validation score: {score:.6f}\n")
    #     output_file.write(f"Tested parameters: {exp}\n")

    print(f"\n=== Trial {feedback.id} ===\n")
    print(f"Validation score: {score:.6f}\n")
    print(f"Tested parameters: {exp}\n")

    # Track best
    if best_score is None or score > best_score:
        best_score = score
        best_test_probs = test_probs
        best_exp = exp

    score = float("-inf")
    run += 1
    # trial += 1 if i > 64 else 0  # Warm-up phase does not count towards trial count

print(f"\n=== Search Complete ===")
print(f"Best Validation Score: {best_score:.6f}")
print(f"Best Parameters: {best_exp}")

# result = run_with_timeout(exp_template.run, timeout_sec=_timeout)
# success, (best_score, best_test_probs) = result
# print(f"Best Validation Score: {best_score:.6f}")

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