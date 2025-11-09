import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import wandb, os
os.environ['WANDB_API_KEY'] = '51e19a7c4d2d6d577fd64d1d9e64a43fa83ccafa'
# Disable WandB console output to avoid TTY issues
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['WANDB_SILENT'] = 'true'

# Initialize Weights & Biases
wandb.init(
    project="spaceship-titanic",
    name="human-catboost-wandb",
    config={
        "model": "CatBoostClassifier",
        "depth": 10,
        "loss_function": "Logloss",
        "eval_metric": "Accuracy",
        # "learning_rate": 0.03,
    },
    settings=wandb.Settings(console="off", silent=True),
)

# Define custom x-axes for curves
wandb.define_metric("iteration")
wandb.define_metric("train/*", step_metric="iteration")
wandb.define_metric("val/*", step_metric="iteration")
wandb.define_metric("full/iteration")
wandb.define_metric("full/*", step_metric="full/iteration")

# Custom callback to log metrics to WandB
class WandBCallback:
    def __init__(self, log_every_n=10):
        self.log_every_n = log_every_n
        
    def after_iteration(self, info):
        iteration = info.iteration
        
        # Log every n iterations
        if iteration % self.log_every_n == 0:
            metrics = info.metrics
            log_dict = {"iteration": iteration}
            
            # Log training loss
            if "learn" in metrics and "Logloss" in metrics["learn"]:
                log_dict["train/loss"] = metrics["learn"]["Logloss"][-1]
            
            # Log validation accuracy (if validation set provided)
            if "validation" in metrics and "Accuracy" in metrics["validation"]:
                log_dict["val/accuracy"] = metrics["validation"]["Accuracy"][-1]
            
            # # Log learning rate (constant in basic CatBoost)
            # log_dict["train/lr"] = wandb.config.learning_rate
            
            wandb.log(log_dict)
        
        return True 
    
def _lr_list(model, n):
    lr = model.learning_rate_
    if isinstance(lr, (list, tuple, np.ndarray)):
        vals = list(lr)[:n]
        if len(vals) < n:
            vals += [vals[-1]] * (n - len(vals))
        return vals
    return [float(lr)] * n

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt


# import data
df_train_orig = pd.read_csv("./input/train.csv")
df_test_orig = pd.read_csv("./input/test.csv")

df_train_orig


df_train_orig.info()


def fill_df(df):
    obj_cols = ["PassengerId","HomePlanet","CryoSleep","Cabin","Destination","VIP","Name"]
    for i in df.columns:
        if i in obj_cols:
            df[i] = df[i].fillna(df[i].mode()[0])
        else:
            df[i] = df[i].fillna(df[i].median())
    return df


cat_train = fill_df(df_train_orig)
cat_test = fill_df(df_test_orig)


cat_train["CryoSleep"] = cat_train["CryoSleep"].astype(int)
cat_train["VIP"] = cat_train["VIP"].astype(int)

cat_test["CryoSleep"] = cat_test["CryoSleep"].astype(int)
cat_test["VIP"] = cat_test["VIP"].astype(int)


tmp_df = cat_train["Cabin"].str.split("/",expand=True)
tmp_df.columns = ["Deck","Num","Side"]

tmp_df_test = cat_test["Cabin"].str.split("/",expand=True)
tmp_df_test.columns = ["Deck","Num","Side"]

cat_train = pd.concat([cat_train,tmp_df],axis=1)
cat_train = cat_train.drop('Cabin',axis=1)

cat_test = pd.concat([cat_test,tmp_df_test],axis=1)
cat_test = cat_test.drop('Cabin',axis=1)


cat_train = cat_train.drop(["PassengerId","Name"],axis=1)
cat_test = cat_test.drop(["PassengerId","Name"],axis=1)


y_cat_train = cat_train["Transported"]
cat_train = cat_train.drop(["Transported"],axis=1)


y_cat_train = y_cat_train.astype(int)


cat_train['Num'] = cat_train['Num'].astype(int)
cat_test['Num'] = cat_test['Num'].astype(int)


 


X_train, X_test, y_train, y_test = train_test_split(cat_train, y_cat_train, test_size=0.2, random_state=0)


X_train.info()

print("test 1")

from catboost import CatBoostClassifier

model = CatBoostClassifier(depth=10,
                           loss_function='Logloss',
                           eval_metric = 'Accuracy', 
                           cat_features=["HomePlanet","Deck","Destination","Side"])


# Train with validation to get curves
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True,
    verbose=False
)

print("test 2")

# Log per-iteration training loss, validation accuracy, and LR
evals = model.get_evals_result()
learn_loss = evals.get("learn", {}).get("Logloss", [])
val_acc = evals.get("validation", {}).get("Accuracy", [])
n_iter = len(learn_loss)
lr_vals = _lr_list(model, n_iter)
for i in range(n_iter):
    payload = {
        "iteration": i + 1,
        "train/loss": learn_loss[i],
        "train/lr": lr_vals[i],
    }
    if i < len(val_acc):
        payload["val/accuracy"] = val_acc[i]
    wandb.log(payload)

y_pred = model.predict(X_test)

print("test 3")

y_pred = pd.Series(y_pred)
final_accuracy = accuracy_score(y_test,y_pred)

wandb.log({
    "train/test_accuracy": final_accuracy
})
print("W&B URL:", getattr(wandb.run, "url", None))

# Full-data stage: retrain with best_iteration on all training data
model = CatBoostClassifier(depth=10,
                           loss_function='Logloss',
                           eval_metric = 'Accuracy', 
                           cat_features=["HomePlanet","Deck","Destination","Side"])

model.fit(X_train, y_train, callbacks=[WandBCallback(log_every_n=10)])

print("test 4")

full_evals = model.get_evals_result()
full_learn_loss = full_evals.get("learn", {}).get("Logloss", [])
full_n_iter = len(full_learn_loss)
full_lr_vals = _lr_list(model, full_n_iter)

for i in range(full_n_iter):
    wandb.log({
        "full/iteration": i + 1,
        "full/loss": full_learn_loss[i],
        "full/lr": full_lr_vals[i],
        # Gradient norms are not exposed by CatBoost; cannot log train/grad_norm or full/grad_norm
    })

print("test 5")

y_pred = model.predict(X_test)


y_pred = pd.Series(y_pred)
final_accuracy = accuracy_score(y_test,y_pred)

wandb.log({
    "final/test_accuracy": final_accuracy
})

print("test 6")

y_pred = model.predict(cat_test)
y_pred = pd.Series(y_pred)
y_pred = y_pred.astype(bool)


df = pd.DataFrame({"PassengerId":df_test_orig["PassengerId"],"Transported":y_pred})
df


df.to_csv("submission/submission.csv",index=False)

wandb.finish()