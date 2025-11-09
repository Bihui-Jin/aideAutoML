# jupyter nbconvert --to script --no-prompt  /home/b27jin/mle-bench-internal/docker-test/scripts_full/playground-series-s3e18_benidictusgalihmp_ps3e18-xgboost_v17_C1.ipynb --output-dir /home/b27jin/aideAutoML/verification/playground-series-s3e18/human.py

import wandb, os
os.environ['WANDB_API_KEY'] = '51e19a7c4d2d6d577fd64d1d9e64a43fa83ccafa'
# Disable WandB console output to avoid TTY issues
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['WANDB_SILENT'] = 'true'

# Initialize Weights & Biases
wandb.init(
    project="playground-series-s3e18",
    name=f"human-xgboost-{os.getpid()}",
    config={
        "model": "XGBClassifier",
        # "depth": 10,
        "eta": 0.3,
        "max_depth": 3,
        # "learning_rate": 0.03,
    },
    settings=wandb.Settings(console="off", silent=True),
)

# Define custom x-axes for curves
# wandb.define_metric("feature")
# wandb.define_metric("iteration")
# wandb.define_metric("train/*", step_metric="iteration")
# wandb.define_metric("val/*", step_metric="iteration")

wandb.define_metric("train/EC1_iteration")
wandb.define_metric("train/EC1_*", step_metric="train/EC1_iteration")
wandb.define_metric("val/EC1_*", step_metric="train/EC1_iteration")

wandb.define_metric("train/EC2_iteration")
wandb.define_metric("train/EC2_*", step_metric="train/EC2_iteration")
wandb.define_metric("val/EC2_*", step_metric="train/EC2_iteration")

wandb.define_metric("full/EC1_iteration")
wandb.define_metric("full/EC1_*", step_metric="full/EC1_iteration")

wandb.define_metric("full/EC2_iteration")
wandb.define_metric("full/EC2_*", step_metric="full/EC2_iteration")


    
def _lr_list(model, n):
    try:
        lr = model.get_xgb_params().get('learning_rate', None)
    except Exception:
        lr = None
    try:
        return [float(lr)] * n
    except Exception:
        return [np.nan] * n

import pandas as pd
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


#Daten laden
df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')
submission = pd.read_csv('./input/sample_submission.csv', index_col="id")

#Trainings- und Testdaten vorbereiten
# y_train1 = df_train['EC1']
# y_train2 = df_train['EC2']
X_train = df_train.drop(['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6'], axis=1)

TARGETS = ["EC1", "EC2"]
iter_offset = 0
for target in TARGETS:
    y_train = df_train[target]
    aucs = []

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size= 0.2,
        random_state=42,
        stratify=y_train
    )

    model = XGBClassifier(eta=0.03, max_depth=3)

    model.fit(X_tr, y_tr,
              eval_set=[(X_tr, y_tr), (X_val, y_val)],
              verbose=False,
    )

    evals = model.evals_result()
    keys = list(evals.keys())  # ["validation_0", "validation_1"]
    train_key = keys[0]
    val_key = keys[1] if len(keys) > 1 else None

    learn_loss = [float(x) for x in evals.get(train_key, {}).get("logloss", [])]
    val_acc = [float(x) for x in evals.get(val_key, {}).get("accuracy", [])] if val_key else []
    n_iter = len(learn_loss)
    lr_vals = _lr_list(model, n_iter)

    for i in range(n_iter):
        step = iter_offset + i + 1
        payload = {
            f"train/{target}_iteration": i + 1,
            f"train/{target}_loss": learn_loss[i],
            f"train/{target}_lr": lr_vals[i],
        }
        if i < len(val_acc):
            payload[f"val/{target}_accuracy"] = val_acc[i]
        wandb.log(payload)

    try:
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        aucs.append(auc)
        wandb.log({f"val/auc_{target}_mean": float(np.mean(aucs))})
    except Exception:
        pass

    model = XGBClassifier(eta=0.03, max_depth=3)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    evals = model.evals_result()
    keys = list(evals.keys())  # ["validation_0", "validation_1"]
    train_key = keys[0]
    # val_key = keys[1] if len(keys) > 1 else None

    learn_loss = [float(x) for x in evals.get(train_key, {}).get("logloss", [])]
    n_iter = len(learn_loss)
    lr_vals = _lr_list(model, n_iter)
    for i in range(n_iter):
        wandb.log({
            f"full/{target}_iteration": i + 1,
            f"full/{target}_loss": learn_loss[i],
            f"full/{target}_lr": lr_vals[i],
        })

    submission[target] = model.predict_proba(df_test)[:, 1]
    iter_offset += n_iter

# model1 = XGBClassifier(eta=0.03, max_depth=3)
# model2 = XGBClassifier(eta=0.03, max_depth=3)

# model1.fit(X_train, y_train1)
# model2.fit(X_train, y_train2)


# y1_predict = model1.predict_proba(df_test)
# y2_predict = model2.predict_proba(df_test)


# df_submit = pd.read_csv('./input/sample_submission.csv', index_col='id')
# df_submit['EC1'] = y1_predict[:,1]
# df_submit['EC2'] = y2_predict[:,1]
submission.to_csv('submission/submission.csv')
wandb.finish()