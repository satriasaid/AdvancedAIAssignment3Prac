import os
import torch
import optuna
from torch.utils.data import DataLoader
import ecg_module

# Override directories for Optuna training — combined PTB-XL + Challenge 2020
BASE_DIR = "/Users/satriasaid/Library/CloudStorage/OneDrive-Curtin/Units/Semester 3 2026/Advanced Artificial Intelligence Research Topics/Assignment 3/practical"
PTB_XL_DIR = os.path.join(BASE_DIR, "physionet.org/files/ptb-xl/1.0.3")
CHALLENGE_2020_DIR = os.path.join(BASE_DIR, "physionet.org/files/challenge-2020/1.0.2")

# Set up global config so the datasets can use it
ecg_module.config = ecg_module.ExperimentConfig()
ecg_module.config.ptb_xl_dir = PTB_XL_DIR
ecg_module.config.challenge_2020_dir = CHALLENGE_2020_DIR

print("Loading combined dataset (PTB-XL + Challenge 2020)...")
train_dataset = ecg_module.CombinedECGDataset(
    ptb_xl_dir=PTB_XL_DIR,
    challenge_2020_dir=CHALLENGE_2020_DIR,
    split="train",
    sampling_rate=100,
    target_length=1000,
    normalize=True,
    use_stratified_split=True
)

val_dataset = ecg_module.CombinedECGDataset(
    ptb_xl_dir=PTB_XL_DIR,
    challenge_2020_dir=CHALLENGE_2020_DIR,
    split="val",
    sampling_rate=100,
    target_length=1000,
    normalize=True,
    use_stratified_split=True
)

# Compute class weights from the combined training set
class_weights = ecg_module.get_class_weights(train_dataset.labels)

def objective(trial):
    config = ecg_module.ExperimentConfig()
    config.ptb_xl_dir = PTB_XL_DIR
    config.challenge_2020_dir = CHALLENGE_2020_DIR

    # Global training hyperparameters
    config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    config.batch_size = trial.suggest_categorical("batch_size", [32, 64])
    config.num_epochs = 200
    config.early_stopping_patience = 10

    # Model selection
    model_name = trial.suggest_categorical("model_type", [
        "CNN1D", "ResNet1D", "CNNLSTM", "TransformerOnly", "HybridCNNTransformer"
    ])

    # Model-specific hyperparameters
    if model_name == "CNN1D":
        model = ecg_module.CNN1D(num_leads=12, num_classes=7)
    elif model_name == "ResNet1D":
        base_channels = trial.suggest_categorical("resnet_base_channels", [32, 64])
        model = ecg_module.ResNet1D(num_leads=12, num_classes=7, base_channels=base_channels)
    elif model_name == "CNNLSTM":
        hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [128, 256])
        num_layers = trial.suggest_int("lstm_layers", 1, 3)
        model = ecg_module.CNNLSTM(num_leads=12, num_classes=7, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "TransformerOnly":
        embed_dim = trial.suggest_categorical("trans_embed_dim", [128, 256])
        num_heads = trial.suggest_categorical("trans_heads", [4, 8])
        num_layers = trial.suggest_int("trans_layers", 2, 4)
        model = ecg_module.TransformerOnly(num_leads=12, num_classes=7, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
    elif model_name == "HybridCNNTransformer":
        base_channels = trial.suggest_categorical("hybrid_base_channels", [32, 64])
        embed_dim = trial.suggest_categorical("hybrid_embed_dim", [128, 256])
        num_heads = trial.suggest_categorical("hybrid_heads", [4, 8])
        num_layers = trial.suggest_int("hybrid_layers", 1, 3)
        dropout = trial.suggest_float("hybrid_dropout", 0.1, 0.5)
        model = ecg_module.HybridCNNTransformer(
            num_leads=12, num_classes=7, base_channels=base_channels,
            embed_dim=embed_dim, num_heads=num_heads,
            num_transformer_layers=num_layers, dropout=dropout
        )

    print(f"Trial {trial.number}: model={model_name}, lr={config.learning_rate}, bs={config.batch_size}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Train model
    trained_model, history = ecg_module.train_model(model, train_loader, val_loader, config, class_weights)
    return max(history['val_f1']) if history['val_f1'] else 0.0

if __name__ == "__main__":
    with ecg_module.CarbonTracker(ecg_module.config.output_dir, 'optuna_study') as tracker:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print("\n" + "="*60)
        print("OPTUNA STUDY COMPLETE")
        print("="*60)

        model_types = ["CNN1D", "ResNet1D", "CNNLSTM", "TransformerOnly", "HybridCNNTransformer"]

        for model_name in model_types:
            model_trials = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.params.get("model_type") == model_name
            ]

            if not model_trials:
                print(f"\nNo successful trials for {model_name}. Skipping.")
                continue

            best_trial = max(model_trials, key=lambda t: t.value)
            print(f"\nBest results for {model_name}: F1={best_trial.value:.4f}")

            print(f"Retraining best {model_name}...")
            best_params = best_trial.params

            config = ecg_module.ExperimentConfig()
            config.ptb_xl_dir = PTB_XL_DIR
            config.challenge_2020_dir = CHALLENGE_2020_DIR
            config.learning_rate = best_params["learning_rate"]
            config.weight_decay = best_params["weight_decay"]
            config.batch_size = best_params["batch_size"]
            config.num_epochs = 30

            if model_name == "CNN1D":
                model = ecg_module.CNN1D(num_leads=12, num_classes=7)
            elif model_name == "ResNet1D":
                model = ecg_module.ResNet1D(num_leads=12, num_classes=7, base_channels=best_params["resnet_base_channels"])
            elif model_name == "CNNLSTM":
                model = ecg_module.CNNLSTM(num_leads=12, num_classes=7, hidden_dim=best_params["lstm_hidden_dim"], num_layers=best_params["lstm_layers"])
            elif model_name == "TransformerOnly":
                model = ecg_module.TransformerOnly(num_leads=12, num_classes=7, embed_dim=best_params["trans_embed_dim"], num_heads=best_params["trans_heads"], num_layers=best_params["trans_layers"])
            elif model_name == "HybridCNNTransformer":
                model = ecg_module.HybridCNNTransformer(
                    num_leads=12, num_classes=7, base_channels=best_params["hybrid_base_channels"],
                    embed_dim=best_params["hybrid_embed_dim"], num_heads=best_params["hybrid_heads"],
                    num_transformer_layers=best_params["hybrid_layers"], dropout=best_params["hybrid_dropout"]
                )

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

            trained_model, history = ecg_module.train_model(model, train_loader, val_loader, config, class_weights)

            save_path = os.path.join(config.output_dir, f"best_model_{model_name.lower()}.pth")
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'model_type': model_name,
                'hyperparameters': best_params,
                'f1_macro': max(history['val_f1'])
            }, save_path)

            print(f"Saved best {model_name} to {save_path}")

        print("\nAll best models saved to the output directory.")