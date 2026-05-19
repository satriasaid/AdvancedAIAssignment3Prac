import sys
import platform
import os
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import shap

# Global configuration placeholder (initialized with defaults)
CARBON_TRACKING = True
from codecarbon import EmissionsTracker

@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""
    random_seed: int = 42
    np_seed: int = 42
    torch_seed: int = 42
    base_dir: str = '/home/21522611/Documents/AdvancedAIAssignment3Prac'
    output_dir: str = 'output'
    data_dir: str = 'data'
    ptb_xl_dir: str = 'physionet.org/files/ptb-xl/1.0.3/'
    challenge_2020_dir: str = 'physionet.org/files/challenge-2020/1.0.2'
    ptb_xl_url: str = 'https://physionet.org/files/ptb-xl/1.0.3/'
    challenge_2020_url: str = 'https://physionet.org/files/challenge-2020/1.0.2/'
    validation_dir: str = 'validation'
    test_dir: str = 'test'
    sampling_rate: int = 100
    target_sampling_rate: int = 100
    num_leads: int = 12
    signal_length: int = 1000
    batch_size: int = 32
    num_epochs: int = 50
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_classes: int = 7
    class_names: List[str] = None
    confidence_threshold: float = 0.7
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 0

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB', 'OTHERS']
        self.base_dir = Path(self.base_dir)
        self.output_dir = self.base_dir / self.output_dir
        self.data_dir = self.base_dir / self.data_dir
        self.ptb_xl_dir = self.base_dir / self.ptb_xl_dir
        self.challenge_2020_dir = self.base_dir / self.challenge_2020_dir
        self.validation_dir = self.base_dir / self.validation_dir
        self.test_dir = self.base_dir / self.test_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

# Initialize global config with default values
config = ExperimentConfig()

def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_system_info() -> Dict[str, Any]:
    """Collect system and package version information."""
    info = {'timestamp': datetime.now().isoformat(), 'python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}', 'platform': platform.platform(), 'processor': platform.processor()}
    packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'matplotlib', 'seaborn', 'shap']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            info[f'{pkg}_version'] = mod.__version__
        except (ImportError, AttributeError):
            pass
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
    info['carbon_tracking_enabled'] = CARBON_TRACKING
    return info

class CarbonTracker:
    """Context manager for tracking carbon emissions."""

    def __init__(self, output_dir: Path, experiment_name: str='training'):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.tracker = None
        self.emissions_data = None

    def __enter__(self):
        if CARBON_TRACKING:
            self.tracker = EmissionsTracker(output_dir=str(self.output_dir), output_file=f'carbon_emissions_{self.experiment_name}.csv', log_level='warning')
            self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker:
            self.emissions_data = self.tracker.stop()

    def get_emissions(self) -> Optional[float]:
        if self.emissions_data:
            return self.emissions_data
        return None
import wfdb
import requests
from zipfile import ZipFile
from io import BytesIO

def download_ptb_xl(data_dir: Path, force: bool=False) -> Path:
    """
    Download PTB-XL dataset from PhysioNet.
    """
    ptb_dir = config.ptb_xl_dir if hasattr(config, 'ptb_xl_dir') else data_dir / 'ptb-xl'
    if ptb_dir.exists() and (not force):
        print(f'PTB-XL dataset already exists at {ptb_dir}')
        return ptb_dir
    print('Downloading PTB-XL dataset from PhysioNet...')
    ptb_dir.mkdir(parents=True, exist_ok=True)
    try:
        wfdb.dl_database_files('ptb-xl', str(ptb_dir))
        print(f'PTB-XL dataset downloaded to {ptb_dir}')
    except Exception as e:
        print(f'Error downloading PTB-XL: {e}')
    return ptb_dir

def download_challenge_2020(data_dir: Path, force: bool=False) -> Path:
    """
    Download Challenge 2020 dataset from PhysioNet.
    """
    challenge_dir = config.challenge_2020_dir if hasattr(config, 'challenge_2020_dir') else data_dir / 'challenge-2020'
    if challenge_dir.exists() and (not force):
        print(f'Challenge 2020 dataset already exists at {challenge_dir}')
        return challenge_dir
    print('Downloading Challenge 2020 dataset from PhysioNet...')
    challenge_dir.mkdir(parents=True, exist_ok=True)
    try:
        wfdb.dl_database_files('challenge-2020', str(challenge_dir))
        print(f'Challenge 2020 dataset downloaded to {challenge_dir}')
    except Exception as e:
        print(f'Error downloading Challenge 2020: {e}')
    return challenge_dir

def download_all_datasets(force: bool=False):
    """Download all required datasets."""
    results = {}
    print('=' * 60)
    print('DATASET DOWNLOAD')
    print('=' * 60)
    print('\nNote: Dataset download is disabled by default.')
    print(f'- PTB-XL: {config.ptb_xl_url} -> {config.ptb_xl_dir}')
    print(f'- Challenge 2020: {config.challenge_2020_url} -> {config.challenge_2020_dir}')
    return results
PTB_XL_LABEL_MAP = {'NORM': 'NORM', 'SR': 'NORM', 'AFIB': 'AFIB', 'AFAF': 'AFIB', 'AFLT': 'AFLT', 'STACH': 'AFLT', '1AVB': '1dAVb', 'AVB1': '1dAVb', 'RBBB': 'RBBB', 'IRBBB': 'RBBB', 'LBBB': 'LBBB', 'ILBBB': 'LBBB', 'LBBB1': 'LBBB', 'LBBB2': 'LBBB', 'LBBB3': 'LBBB'}
CHALLENGE_2020_LABEL_MAP = {'426783006': 'AFIB', '164889003': 'AFLT', '164865005': '1dAVb', '713427006': 'RBBB', '713426002': 'LBBB', '270492004': 'NORM', '426177001': 'NORM'}

def map_ptb_xl_label(diagnostic_codes: List[str]) -> str:
    """
    Map PTB-XL diagnostic codes to assignment classes.
    Returns the first matching disease class (as per assignment brief).
    """
    priority_order = ['AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB']
    for code in diagnostic_codes:
        for priority_class in priority_order:
            if code in PTB_XL_LABEL_MAP and PTB_XL_LABEL_MAP[code] == priority_class:
                return priority_class
    for code in diagnostic_codes:
        if code in PTB_XL_LABEL_MAP and PTB_XL_LABEL_MAP[code] == 'NORM':
            has_abnormal = any((c in PTB_XL_LABEL_MAP and PTB_XL_LABEL_MAP[c] != 'NORM' for c in diagnostic_codes))
            if not has_abnormal:
                return 'NORM'
    return 'OTHERS'

def map_challenge_2020_label(diagnostic_codes: List[str]) -> str:
    """Map Challenge 2020 diagnostic codes to assignment classes."""
    priority_order = ['AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB']
    for code in diagnostic_codes:
        if code in CHALLENGE_2020_LABEL_MAP:
            mapped = CHALLENGE_2020_LABEL_MAP[code]
            if mapped in priority_order:
                return mapped
    for code in diagnostic_codes:
        if code in CHALLENGE_2020_LABEL_MAP and CHALLENGE_2020_LABEL_MAP[code] == 'NORM':
            has_abnormal = any((c in CHALLENGE_2020_LABEL_MAP and CHALLENGE_2020_LABEL_MAP[c] != 'NORM' for c in diagnostic_codes))
            if not has_abnormal:
                return 'NORM'
    return 'OTHERS'

class PTBXLDataset(Dataset):
    """PTB-XL ECG Dataset."""

    def __init__(self, data_dir: Path, split: str='train', sampling_rate: int=100, target_length: int=1000, normalize: bool=True, use_stratified_split: bool=True):
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.normalize = normalize
        self.metadata = self._load_metadata()
        if use_stratified_split:
            self.records = self._stratified_split(split)
        else:
            self.records = self._official_split(split)
        
        # Filter out records with missing files to avoid FileNotFoundError
        self.records = self._filter_missing_records(self.records)
        
        self.labels = [self._map_label(rec) for rec in self.records]
        print(f'PTB-XL {split} set: {len(self.records)} records')
        self._print_class_distribution()

    def _filter_missing_records(self, records: List[int]) -> List[int]:
        """Filter out records whose data files are missing from disk."""
        valid_records = []
        missing_count = 0
        for rec_idx in records:
            row = self.metadata.iloc[rec_idx]
            if self.sampling_rate <= 100:
                filename = row['filename_lr']
            else:
                filename = row['filename_hr']
            
            # Check if both .hea and .dat files exist (wfdb needs both)
            hea_path = self.data_dir / (filename + ".hea")
            dat_path = self.data_dir / (filename + ".dat")
            if hea_path.exists() and dat_path.exists():
                valid_records.append(rec_idx)
            else:
                missing_count += 1
        
        if missing_count > 0:
            warnings.warn(f"Filtered out {missing_count} PTB-XL records due to missing files on disk.")
        return valid_records

    def _load_metadata(self) -> pd.DataFrame:
        """Load PTB-XL metadata file."""
        yaml_path = self.data_dir / 'ptbxl_database.csv'
        if not yaml_path.exists():
            raise FileNotFoundError(f'PTB-XL metadata not found at {yaml_path}. Please download the dataset first.')
        df = pd.read_csv(yaml_path)
        df['diagnostic_codes'] = df['scp_codes'].apply(self._parse_scp_codes)
        return df

    def _parse_scp_codes(self, scp_string: str) -> List[str]:
        """Parse SCP codes from string."""
        try:
            import ast
            codes_dict = ast.literal_eval(scp_string)
            return list(codes_dict.keys())
        except:
            return []

    def _map_label(self, record_idx: int) -> str:
        """Map PTB-XL record to assignment label."""
        row = self.metadata.iloc[record_idx]
        return map_ptb_xl_label(row['diagnostic_codes'])

    def _stratified_split(self, split: str) -> List[int]:
        """Create stratified train/val split by patient."""
        patient_ids = self.metadata['patient_id'].unique()
        patient_labels = {}
        for pid in patient_ids:
            patient_records = self.metadata[self.metadata['patient_id'] == pid]
            patient_labels[pid] = self._map_label(patient_records.index[0])
        (train_pids, val_pids) = train_test_split(list(patient_ids), test_size=0.2, random_state=config.random_seed, stratify=[patient_labels[pid] for pid in patient_ids])
        if split == 'train':
            selected_pids = train_pids
        else:
            selected_pids = val_pids
        records = self.metadata[self.metadata['patient_id'].isin(selected_pids)].index.tolist()
        return records

    def _official_split(self, split: str) -> List[int]:
        """Use official PTB-XL splits (stratified by 10-fold)."""
        if split == 'train':
            folds = [1, 2, 3, 4, 5, 6, 7, 8]
        elif split == 'val':
            folds = [9]
        else:
            folds = [10]
        return self.metadata[self.metadata['strat_fold'].isin(folds)].index.tolist()

    def _print_class_distribution(self):
        """Print class distribution."""
        from collections import Counter
        counts = Counter(self.labels)
        print(f'  Class distribution: {dict(counts)}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_idx = self.records[idx]
        row = self.metadata.iloc[record_idx]
        if self.sampling_rate <= 100:
            filename = row['filename_lr']
        else:
            filename = row['filename_hr']
        filepath = self.data_dir / filename
        record_path = str(filepath).replace('.dat', '').replace('.hea', '')
        record = wfdb.rdrecord(record_path)
        ecg = record.p_signal.T
        ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        if record.fs != self.sampling_rate:
            num_samples = int(ecg.shape[1] * self.sampling_rate / record.fs)
            ecg = signal.resample(ecg, num_samples, axis=1)
        ecg = self._pad_or_truncate(ecg)
        if self.normalize:
            ecg = self._normalize(ecg)
        label = self.labels[idx]
        label_idx = config.class_names.index(label)
        return {'ecg': torch.FloatTensor(ecg), 'label': torch.LongTensor([label_idx]), 'label_str': label, 'record_id': str(record_idx), 'source': 'ptb_xl'}

    def _pad_or_truncate(self, ecg: np.ndarray) -> np.ndarray:
        """Pad or truncate ECG to target length."""
        if ecg.shape[1] < self.target_length:
            padding = self.target_length - ecg.shape[1]
            ecg = np.pad(ecg, ((0, 0), (0, padding)), mode='constant')
        elif ecg.shape[1] > self.target_length:
            ecg = ecg[:, :self.target_length]
        return ecg

    def _normalize(self, ecg: np.ndarray) -> np.ndarray:
        """Z-score normalize per lead."""
        mean = ecg.mean(axis=1, keepdims=True)
        std = ecg.std(axis=1, keepdims=True)
        std[std < 1e-08] = 1.0
        return (ecg - mean) / std

class Challenge2020Dataset(Dataset):
    """Challenge 2020 ECG Dataset — loads directly from wfdb database files.

    Records are downloaded on-the-fly (via wfdb) if the local directory is empty.
    """

    def __init__(self, data_dir: Path, sampling_rate: int = 100, target_length: int = 1000, normalize: bool = True, force_download: bool = False):
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.normalize = normalize

        # Find records: check subfolders recursively for .hea files
        hea_files = sorted(self.data_dir.glob("**/*.hea"))
        if len(hea_files) == 0:
            if force_download or not (self.data_dir / "RECORDS").exists():
                print(f"No Challenge 2020 records found in {self.data_dir}.")
                print("Attempting to download via wfdb...")
                self._download_via_wfdb()
                hea_files = sorted(self.data_dir.glob("**/*.hea"))

        if len(hea_files) == 0:
            raise FileNotFoundError(
                f"No Challenge 2020 records (.hea files) found in {self.data_dir}. "
                "Please download the dataset: wfdb.dl_database_files('challenge-2020', <path>)"
            )

        # Extract record paths (relative to data_dir, without .hea extension)
        all_records = sorted([str(f.relative_to(self.data_dir)).replace(".hea", "") for f in hea_files])
        
        # Filter out records missing .mat data files
        self.records = []
        missing_count = 0
        for rec in all_records:
            mat_path = self.data_dir / (rec + ".mat")
            if mat_path.exists():
                self.records.append(rec)
            else:
                missing_count += 1
        if missing_count > 0:
            warnings.warn(f"Filtered out {missing_count} Challenge 2020 records due to missing .mat files.")
        
        self._load_reference_data()
        self.labels = [self._map_label(rec) for rec in self.records]
        print(f"Challenge 2020 set: {len(self.records)} records")
        self._print_class_distribution()

    def _download_via_wfdb(self):
        """Download Challenge 2020 via wfdb."""
        try:
            wfdb.dl_database_files("challenge-2020", str(self.data_dir))
            print(f"Downloaded Challenge 2020 to {self.data_dir}")
        except Exception as e:
            print(f"wfdb download failed: {e}")
            raise FileNotFoundError(
                f"Challenge 2020 download failed. Please manually download from "
                f"https://physionet.org/files/challenge-2020/1.0.2/ into {self.data_dir}"
            )

    def _load_reference_data(self):
        """Load Challenge 2020 reference CSV."""
        import ast
        ref_path = self.data_dir / "reference.csv"
        self.reference = {}
        if ref_path.exists():
            df = pd.read_csv(ref_path)
            for _, row in df.iterrows():
                self.reference[row["record"]] = row
        else:
            warnings.warn(f"reference.csv not found at {ref_path}. Using first-label mapping.")

    def _parse_labels_csv(self, labels_str: str) -> List[str]:
        """Parse labels from CSV string."""
        try:
            labels = ast.literal_eval(labels_str)
            if isinstance(labels, list):
                return labels
            return [labels]
        except:
            return []

    def _map_label(self, record_path: str) -> str:
        """Map Challenge 2020 record to assignment label."""
        record_stem = Path(record_path).stem
        if record_stem in self.reference:
            row = self.reference[record_stem]
            if "labels" in row:
                diagnostic_codes = self._parse_labels_csv(str(row["labels"]))
                return map_challenge_2020_label(diagnostic_codes)
        
        # Fallback: Try to read labels from the .hea file directly
        try:
            hea_file = self.data_dir / (record_path + ".hea")
            if hea_file.exists():
                with open(hea_file, 'r') as f:
                    for line in f:
                        if line.startswith('# Dx:') or line.startswith('#Dx:'):
                            diagnostic_codes = line.split(': ')[1].strip().split(',')
                            return map_challenge_2020_label(diagnostic_codes)
        except Exception:
            pass
            
        return "OTHERS"

    def _print_class_distribution(self):
        """Print class distribution."""
        from collections import Counter
        counts = Counter(self.labels)
        print(f"  Class distribution: {dict(counts)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_name = self.records[idx]
        # wfdb reads the record by name; it looks for .hea/.dat alongside each other
        record_path = str(self.data_dir / record_name)
        try:
            record = wfdb.rdrecord(record_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load record {record_name}: {e}")
        ecg = record.p_signal.T  # Shape: (num_leads, samples)
        ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

        # Resample if needed
        if record.fs != self.sampling_rate:
            num_samples = int(ecg.shape[1] * self.sampling_rate / record.fs)
            ecg = signal.resample(ecg, num_samples, axis=1)

        ecg = self._pad_or_truncate(ecg)
        if self.normalize:
            ecg = self._normalize(ecg)

        label = self.labels[idx]
        label_idx = config.class_names.index(label)
        return {
            "ecg": torch.FloatTensor(ecg),
            "label": torch.LongTensor([label_idx]),
            "label_str": label,
            "record_id": record_name,
            "source": "challenge_2020",
        }

    def _pad_or_truncate(self, ecg: np.ndarray) -> np.ndarray:
        if ecg.shape[1] < self.target_length:
            ecg = np.pad(ecg, ((0, 0), (0, self.target_length - ecg.shape[1])), mode="constant")
        elif ecg.shape[1] > self.target_length:
            ecg = ecg[:, : self.target_length]
        return ecg

    def _normalize(self, ecg: np.ndarray) -> np.ndarray:
        mean = ecg.mean(axis=1, keepdims=True)
        std = ecg.std(axis=1, keepdims=True)
        std[std < 1e-08] = 1.0
        return (ecg - mean) / std


class CombinedECGDataset(Dataset):
    """
    Combined dataset wrapping PTB-XL and Challenge 2020 for joint training.

    Both datasets are loaded, then split together using PTB-XL's stratified
    patient-level split as the primary split, with Challenge 2020 records
    added to the train split only.
    """

    def __init__(
        self,
        ptb_xl_dir: Path,
        challenge_2020_dir: Path,
        split: str = "train",
        sampling_rate: int = 100,
        target_length: int = 1000,
        normalize: bool = True,
        use_stratified_split: bool = True,
        val_ratio: float = 0.2,
        challenge_as_val: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.normalize = normalize

        # Load PTB-XL dataset
        self.ptb_dataset = PTBXLDataset(
            data_dir=ptb_xl_dir,
            split=split,
            sampling_rate=sampling_rate,
            target_length=target_length,
            normalize=normalize,
            use_stratified_split=use_stratified_split,
        )

        # Load Challenge 2020 dataset
        try:
            self.challenge_dataset = Challenge2020Dataset(
                data_dir=challenge_2020_dir,
                sampling_rate=sampling_rate,
                target_length=target_length,
                normalize=normalize,
            )
            has_challenge = True
        except Exception as e:
            warnings.warn(f"Could not load Challenge 2020 dataset: {e}")
            self.challenge_dataset = None
            has_challenge = False

        # Build combined index
        if split == "train":
            if has_challenge and not challenge_as_val:
                # Add all Challenge 2020 records to training
                self._indices = [
                    (self.ptb_dataset, i) for i in range(len(self.ptb_dataset))
                ] + [
                    (self.challenge_dataset, i) for i in range(len(self.challenge_dataset))
                ]
                print(f"Combined train set: {len(self.ptb_dataset)} PTB-XL + {len(self.challenge_dataset)} Challenge 2020 = {len(self)} total")
            else:
                self._indices = [(self.ptb_dataset, i) for i in range(len(self.ptb_dataset))]
                print(f"Combined train set: {len(self.ptb_dataset)} PTB-XL = {len(self)} total")
        else:
            # Val split: only PTB-XL (Challenge 2020 as val only if explicitly requested)
            self._indices = [(self.ptb_dataset, i) for i in range(len(self.ptb_dataset))]
            print(f"Val set: {len(self.ptb_dataset)} PTB-XL records")

        # Collect all labels for class-weight computation
        self.labels = [self._get_label(idx) for idx in range(len(self))]
        self._print_class_distribution()

    def _get_label(self, idx):
        ds, i = self._indices[idx]
        return ds.labels[i]

    def _print_class_distribution(self):
        from collections import Counter
        counts = Counter(self.labels)
        print(f"  Combined class distribution: {dict(counts)}")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        ds, i = self._indices[idx]
        return ds[i]


class LocalECGDataset(Dataset):
    """Dataset for local validation/test ECG files."""

    def __init__(self, data_dir: Path, normalize: bool=True, has_labels: bool=True):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.has_labels = has_labels
        self.files = sorted(list(self.data_dir.glob('*/*.npy')))
        if not self.files:
            raise FileNotFoundError(f'No .npy files found in {data_dir}')
        self.labels = []
        self.filenames = []
        for f in self.files:
            self.filenames.append(f.stem)
            if has_labels:
                parts = f.stem.split('-')
                if len(parts) > 1:
                    label = parts[-1].replace('.png', '')
                    if label in config.class_names:
                        self.labels.append(label)
                    else:
                        self.labels.append('OTHERS')
                else:
                    self.labels.append('OTHERS')
        print(f'Loaded {len(self.files)} ECG files from {data_dir}')
        if has_labels:
            from collections import Counter
            counts = Counter(self.labels)
            print(f'  Class distribution: {dict(counts)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        ecg = np.load(filepath)
        ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        if ecg.shape != (12, 1000):
            if ecg.shape[0] == 1000 and ecg.shape[1] == 12:
                ecg = ecg.T
            else:
                raise ValueError(f'Unexpected ECG shape: {ecg.shape}')
        if self.normalize:
            mean = ecg.mean(axis=1, keepdims=True)
            std = ecg.std(axis=1, keepdims=True)
            std[std < 1e-08] = 1.0
            ecg = (ecg - mean) / std
        output = {'ecg': torch.FloatTensor(ecg), 'filename': self.filenames[idx]}
        if self.has_labels:
            label = self.labels[idx]
            label_idx = config.class_names.index(label)
            output['label'] = torch.LongTensor([label_idx])
            output['label_str'] = label
        return output

def plot_ecg_sample(ecg: np.ndarray, title: str='ECG Sample', leads: List[str]=None, save_path: str=None):
    """Plot 12-lead ECG sample."""
    if leads is None:
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    (fig, axes) = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for (i, lead) in enumerate(leads):
        ax = axes[i // 4, i % 4]
        ax.plot(ecg[i], linewidth=0.5, color='steelblue')
        ax.set_title(lead, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()

def plot_class_distribution(labels: List[str], title: str='Class Distribution', save_path: str=None):
    """Plot class distribution."""
    from collections import Counter
    counts = Counter(labels)
    classes = list(counts.keys())
    values = list(counts.values())
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))
    bars = ax1.bar(classes, values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title(title)
    ax1.tick_params(axis='x', rotation=45)
    for (bar, val) in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val), ha='center', va='bottom', fontsize=9)
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    (wedges, texts, autotexts) = ax2.pie(values, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Percentage Distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()
    return counts

def compute_signal_statistics(dataset: Dataset) -> pd.DataFrame:
    """Compute signal statistics across dataset."""
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    all_ecgs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        all_ecgs.append(sample['ecg'].numpy())
    all_ecgs = np.array(all_ecgs)
    stats = []
    for (i, lead) in enumerate(lead_names):
        lead_data = all_ecgs[:, i, :].flatten()
        stats.append({'Lead': lead, 'Mean': lead_data.mean(), 'Std': lead_data.std(), 'Min': lead_data.min(), 'Max': lead_data.max(), 'Median': np.median(lead_data), 'Range': lead_data.max() - lead_data.min()})
    return pd.DataFrame(stats)

def plot_lead_correlation(dataset: Dataset, title: str='Lead Correlation Matrix', save_path: str=None):
    """Plot correlation matrix between leads."""
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    all_ecgs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        all_ecgs.append(sample['ecg'].numpy())
    all_ecgs = np.array(all_ecgs)
    lead_features = all_ecgs.reshape(len(all_ecgs), 12, -1)
    lead_features = lead_features.reshape(len(all_ecgs), 12, -1).mean(axis=2)
    correlation_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            lead_i = all_ecgs[:, i, :].flatten()
            lead_j = all_ecgs[:, j, :].flatten()
            correlation_matrix[i, j] = np.corrcoef(lead_i, lead_j)[0, 1]
    (fig, ax) = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation_matrix, cmap='Spectral_r', vmin=-1, vmax=1)
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(lead_names)
    ax.set_yticklabels(lead_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    ax.set_title(title, fontweight='bold')
    for i in range(12):
        for j in range(12):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()

class ConvBlock1D(nn.Module):
    """1D Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock1D(nn.Module):
    """Residual block for 1D signals with optional downsampling and channel change."""

    def __init__(self, in_channels, out_channels=None, kernel_size=7, stride=1, dilation=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
            
        padding = (kernel_size - 1) // 2 * dilation
        # Only apply stride to the first convolution
        self.conv1 = ConvBlock1D(in_channels, out_channels, kernel_size, stride, padding, dilation)
        # Second convolution always has stride 1
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return F.relu(out)

class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signals."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2)
        s = F.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s.unsqueeze(2)

class CNN1D(nn.Module):
    """Simple 1D CNN for ECG classification."""

    def __init__(self, num_leads=12, num_classes=7, signal_length=1000):
        super().__init__()
        self.features = nn.Sequential(ConvBlock1D(num_leads, 64, kernel_size=7, stride=2, padding=3), nn.MaxPool1d(2), ConvBlock1D(64, 128, kernel_size=7, stride=2, padding=3), nn.MaxPool1d(2), ConvBlock1D(128, 256, kernel_size=5, stride=2, padding=2), nn.MaxPool1d(2), ConvBlock1D(256, 512, kernel_size=5, stride=1, padding=2), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

class ResNet1D(nn.Module):
    """ResNet-style 1D CNN for ECG classification."""

    def __init__(self, num_leads=12, num_classes=7, base_channels=64):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock1D(num_leads, base_channels, kernel_size=7, stride=2, padding=3), nn.MaxPool1d(2))
        self.stage1 = self._make_stage(base_channels, base_channels, 2, stride=1)
        self.stage2 = self._make_stage(base_channels, base_channels * 2, 2, stride=2)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, 2, stride=2)
        self.stage4 = self._make_stage(base_channels * 4, base_channels * 8, 2, stride=2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(base_channels * 8, num_classes))

    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid for temporal modeling."""

    def __init__(self, num_leads=12, num_classes=7, hidden_dim=256, num_layers=2, bidirectional=True):
        super().__init__()
        self.cnn = nn.Sequential(ConvBlock1D(num_leads, 64, kernel_size=7, stride=2, padding=3), ConvBlock1D(64, 128, kernel_size=5, stride=2, padding=2), ConvBlock1D(128, 256, kernel_size=5, stride=2, padding=2))
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(lstm_output_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0, 2, 1)
        (lstm_out, _) = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        return self.classifier(lstm_out)

class PatchEmbedding1D(nn.Module):
    """Convert 1D signal to patch embeddings."""

    def __init__(self, num_leads=12, patch_size=50, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_leads = num_leads
        self.proj = nn.Linear(num_leads * patch_size, embed_dim)

    def forward(self, x):
        B = x.shape[0]
        num_patches = x.shape[2] // self.patch_size
        x = x[:, :, :num_patches * self.patch_size]
        x = x.reshape(B, self.num_leads, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, num_patches, -1)
        return self.proj(x)

class TransformerEncoder1D(nn.Module):
    """Transformer encoder for 1D signals."""

    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.transformer(x)

class TransformerOnly(nn.Module):
    """Pure Transformer model for ECG classification."""

    def __init__(self, num_leads=12, num_classes=7, patch_size=50, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(num_leads, patch_size, embed_dim)
        num_patches = 1000 // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.transformer = TransformerEncoder1D(embed_dim, num_heads, num_layers)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture with ResNet backbone.
    
    Architecture:
    1. ResNet-style CNN backbone for local feature extraction
    2. Transformer encoder for global context modeling
    3. Gated fusion mechanism
    4. Classification head with confidence estimation
    """

    def __init__(self, num_leads=12, num_classes=7, base_channels=64, embed_dim=256, num_heads=8, num_transformer_layers=2, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock1D(num_leads, base_channels, kernel_size=7, stride=2, padding=3), nn.MaxPool1d(2))
        self.stage1 = nn.Sequential(*[ResidualBlock1D(base_channels) for _ in range(2)])
        self.stage2 = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels * 2, stride=2),
            ResidualBlock1D(base_channels * 2, base_channels * 2)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock1D(base_channels * 4, base_channels * 4)
        )
        self.local_dim = base_channels * 4
        self.patch_size = 25
        self.num_patches = 125 // self.patch_size
        self.patch_proj = nn.Linear(self.local_dim * self.patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_fc = nn.Sequential(nn.Flatten(), nn.Linear(self.local_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(embed_dim, num_classes))
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, return_features=False):
        B = x.shape[0]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        cnn_features = self.stage3(x)
        local_feat = self.global_pool(cnn_features).squeeze(-1)
        local_feat = self.local_fc(local_feat)
        (B, C, T) = cnn_features.shape
        num_patches = T // self.patch_size
        patches = cnn_features[:, :, :num_patches * self.patch_size]
        patches = patches.reshape(B, C, num_patches, self.patch_size)
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(B, num_patches, -1)
        patch_embeds = self.patch_proj(patches)
        global_feat = self.transformer(patch_embeds)
        global_feat = global_feat.mean(dim=1)
        combined = torch.cat([local_feat, global_feat], dim=1)
        gate_weights = self.gate(combined)
        local_weighted = local_feat * gate_weights
        global_weighted = global_feat * (1 - gate_weights)
        fused = torch.cat([local_weighted, global_weighted], dim=1)
        logits = self.classifier(fused)
        scaled_logits = logits / self.temperature
        if return_features:
            return {'logits': scaled_logits, 'local_features': local_feat, 'global_features': global_feat, 'gate_weights': gate_weights, 'cnn_features': cnn_features}
        return scaled_logits

    def predict_with_confidence(self, x, threshold=0.7):
        """Return predictions with confidence scores."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            (max_probs, preds) = probs.max(dim=1)
            low_confidence = max_probs < threshold
            return {'predictions': preds, 'probabilities': probs, 'max_probabilities': max_probs, 'low_confidence': low_confidence}

def count_parameters(model):
    """Count trainable parameters."""
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def load_deepecg_ssl_pretrained(model, pretrained_path, strict=False):
    """
    Load pretrained weights for DeepECGSSL model.

    Args:
        model: DeepECGSSL model instance
        pretrained_path: Path to pretrained checkpoint (.pth or .pt file)
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        return model

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'encoder.' prefix if present (from finetuning checkpoints)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.', '')] = v
            else:
                new_state_dict[k] = v

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Successfully loaded pretrained weights from {pretrained_path}")

    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        print("Continuing with random initialization...")

    return model

def create_architecture_diagram(save_path=None):
    """Create a text-based architecture diagram."""
    diagram = '\n    ╔════════════════════════════════════════════════════════════════════════════╗\n    ║          Hybrid CNN-Transformer Architecture with ResNet Backbone          ║\n    ╠════════════════════════════════════════════════════════════════════════════╣\n    ║                                                                            ║\n    ║  Input: 12-Lead ECG (B, 12, 1000)                                          ║\n    ║         │                                                                   ║\n    ║         ▼                                                                   ║\n    ║  ┌──────────────────────────────────────────────────────────────────┐     ║\n    ║  │                    ResNet Backbone (Local Features)               │     ║\n    ║  │  ┌──────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐          │     ║\n    ║  │  │ Stem │ → │ Stage 1  │ → │ Stage 2  │ → │ Stage 3  │          │     ║\n    ║  │  │ (64) │   │  (64)    │   │  (128)   │   │  (256)   │          │     ║\n    ║  │  └──────┘   └──────────┘   └──────────┘   └──────────┘          │     ║\n    ║  └──────────────────────────────────────────────────────────────────┘     ║\n    ║         │                                                                   ║\n    ║         ├─────────────────────────────────────────────┐                   ║\n    ║         │                                             │                   ║\n    ║         ▼ (Local Path)                               ▼ (Global Path)      ║\n    ║  ┌─────────────┐                             ┌────────────────┐           ║\n    ║  │ Global Pool │                             │  Patches       │           ║\n    ║  │    (256)    │                             │  Projection    │           ║\n    ║  └──────┬──────┘                             └───────┬────────┘           ║\n    ║         │                                            │                     ║\n    ║         ▼                                            ▼                     ║\n    ║  ┌─────────────┐                             ┌────────────────┐           ║\n    ║  │    FC       │                             │  Transformer   │           ║\n    ║  │   (256)     │                             │  Encoder       │           ║\n    ║  └──────┬──────┘                             │  (2 layers,    │           ║\n    ║         │                                     │   8 heads)     │           ║\n    ║         │                                     └───────┬────────┘           ║\n    ║         │                                             │                     ║\n    ║         │                                      Global Pool (256)           ║\n    ║         │                                             │                     ║\n    ║         └────────────────────┬────────────────────────┘                     ║\n    ║                              │                                            ║\n    ║                              ▼                                            ║\n    ║                    ┌─────────────────┐                                     ║\n    ║                    │  Gated Fusion   │                                     ║\n    ║                    │  (Local ⊙ Gate   │                                     ║\n    ║                    │   + Global ⊙     │                                     ║\n    ║                    │    (1-Gate))     │                                     ║\n    ║                    └────────┬─────────┘                                     ║\n    ║                             │                                              ║\n    ║                             ▼                                              ║\n    ║                    ┌─────────────────┐                                     ║\n    ║                    │ Classification  │                                     ║\n    ║                    │ Head (FC → 7)   │                                     ║\n    ║                    │ + Temp Scaling   │                                     ║\n    ║                    └────────┬─────────┘                                     ║\n    ║                             │                                              ║\n    ║                             ▼                                              ║\n    ║              Output: 7-Class Logits + Confidence Score                    ║\n    ║                                                                            ║\n    ╚════════════════════════════════════════════════════════════════════════════╝\n    '
    if save_path:
        with open(save_path, 'w') as f:
            f.write(diagram)
        print(f'Architecture diagram saved to {save_path}')
    print(diagram)

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        self.early_stop = self.counter >= self.patience
        return self.early_stop

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_class_weights(labels):
    """Compute class weights for imbalanced data."""
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.zeros(num_classes)
    for (i, class_name) in enumerate(config.class_names):
        if class_name in counts:
            weights[i] = total / (num_classes * counts[class_name])
        else:
            weights[i] = 1.0
    # Clip weights to prevent destabilizing gradient spikes on extremely rare classes (e.g. LBBB)
    weights = torch.clamp(weights, max=10.0)
    return weights

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels (np array)
        y_pred: Predicted labels (np array)
        y_prob: Prediction probabilities (np array) - optional for AUROC
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, labels=range(len(config.class_names)), target_names=config.class_names, output_dict=True, zero_division=0)
    for cls in config.class_names:
        if cls in report:
            metrics[f'{cls}_precision'] = report[cls]['precision']
            metrics[f'{cls}_recall'] = report[cls]['recall']
            metrics[f'{cls}_f1'] = report[cls]['f1-score']
            metrics[f'{cls}_support'] = report[cls]['support']
    if y_prob is not None:
        try:
            # Check for NaNs in probabilities
            if np.isnan(y_prob).any():
                print("Warning: y_prob contains NaN values. Skipping AUROC/AUPRC computation.")
                return metrics

            y_true_onehot = np.zeros((len(y_true), len(config.class_names)))
            for (i, label) in enumerate(y_true):
                y_true_onehot[i, label] = 1
            auroc_scores = []
            for i in range(len(config.class_names)):
                if y_true_onehot[:, i].sum() > 0:
                    auroc = roc_auc_score(y_true_onehot[:, i], y_prob[:, i])
                    auroc_scores.append(auroc)
            if auroc_scores:
                metrics['auroc_macro'] = np.mean(auroc_scores)
            auprc_scores = []
            for i in range(len(config.class_names)):
                if y_true_onehot[:, i].sum() > 0:
                    (precision, recall, _) = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
                    auprc = auc(recall, precision)
                    auprc_scores.append(auprc)
            if auprc_scores:
                metrics['auprc_macro'] = np.mean(auprc_scores)
        except Exception as e:
            print(f'Warning: Could not compute AUROC/AUPRC: {e}')
    return metrics

def compute_confidence_metrics(probabilities, threshold=0.7):
    """
    Compute confidence-related metrics.
    
    Args:
        probabilities: Prediction probabilities (N, num_classes)
        threshold: Confidence threshold for "low confidence" flag
    
    Returns:
        Dictionary of confidence metrics
    """
    max_probs = probabilities.max(axis=1)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    return {'mean_confidence': max_probs.mean(), 'std_confidence': max_probs.std(), 'min_confidence': max_probs.min(), 'median_confidence': np.median(max_probs), 'low_confidence_ratio': (max_probs < threshold).mean(), 'mean_entropy': entropy.mean(), 'high_entropy_ratio': (entropy > 1.0).mean()}

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional Automatic Mixed Precision (AMP)."""
    model.train()
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    pbar = tqdm(dataloader, desc='Training')
    
    use_amp = (device.type == 'cuda') and (scaler is not None)
    
    for batch in pbar:
        ecg = batch['ecg'].to(device)
        labels = batch['label'].reshape(-1).to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(ecg)
                loss = criterion(outputs, labels)
        else:
            outputs = model(ecg)
            loss = criterion(outputs, labels)
        
        # Check for NaNs/Infs in the loss
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss ({loss.item()}) detected! skipping backward and optimizer step.")
            optimizer.zero_grad()
            continue
            
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping to prevent NaNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # PyTorch's scaler.step internally checks for NaNs/Infs and skips optimizer.step() if found.
            # Calling scaler.update() resets the scaler state, preventing unscale_ errors.
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Check for NaNs/Infs in gradients (only needed when not using AMP)
            gradients_finite = True
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        gradients_finite = False
                        break
            
            if gradients_finite:
                # Gradient clipping to prevent NaNs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                print("Warning: Non-finite gradients detected! skipping optimizer step.")
                optimizer.zero_grad()
        losses.update(loss.item(), ecg.size(0))
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = losses.avg
    return metrics

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            ecg = batch['ecg'].to(device)
            labels = batch['label'].reshape(-1).to(device)
            outputs = model(ecg)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), ecg.size(0))
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = losses.avg
    metrics['confidence'] = compute_confidence_metrics(all_probs, config.confidence_threshold)
    return (metrics, all_labels, all_preds, all_probs)

def train_model(model, train_loader, val_loader, config, class_weights=None, trial=None):
    """
    Full training loop with early stopping, learning rate scheduling, optional AMP, and Optuna pruning.
    
    Returns:
        Trained model, training history
    """
    device = torch.device(config.device)
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode='max')
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': [], 'lr': []}
    best_model_state = None
    best_f1 = 0.0
    
    # Initialize AMP GradScaler if on CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    print(f'\nTraining on {device}')
    print(f'Epochs: {config.num_epochs}')
    print(f'Batch size: {config.batch_size}')
    print(f'Learning rate: {config.learning_rate}')
    if class_weights is not None:
        print(f'Class weights: {class_weights}')
    print('-' * 60)
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch + 1}/{config.num_epochs}')
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        (val_metrics, _, _, _) = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['loss'])
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print(f"Train Loss: {train_metrics['loss']:.4f} | F1: {train_metrics['f1_macro']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1_macro']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            print(f'  ↳ New best F1: {best_f1:.4f}')
            
        # Report progress and check for pruning if running under Optuna trial
        if trial is not None:
            trial.report(val_metrics['f1_macro'], epoch)
            if trial.should_prune():
                print(f"  ↳ Trial pruned early at epoch {epoch + 1} due to poor performance.")
                import optuna
                raise optuna.TrialPruned()
                
        if early_stopping(val_metrics['f1_macro']):
            print(f'\nEarly stopping at epoch {epoch + 1}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model with F1: {best_f1:.4f}')
    return (model, history)

def plot_training_history(history, save_path=None):
    """Plot training history."""
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_f1'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_f1'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score (Macro)')
    axes[1].set_title('Training and Validation F1')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    (fig, ax) = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()
    return cm

def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """Plot ROC curves for each class."""
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    (fig, ax) = plt.subplots(figsize=(10, 8))
    for (i, class_name) in enumerate(class_names):
        if y_true_bin[:, i].sum() > 0:
            (fpr, tpr, _) = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()

def benchmark_model(model_class, model_name, train_loader, val_loader, config, class_weights=None, epochs=20):
    """
    Benchmark a single model.
    
    Returns:
        Model, history, metrics
    """
    print(f"\n{'=' * 60}")
    print(f'Training: {model_name}')
    print(f"{'=' * 60}")
    model = model_class()
    original_epochs = config.num_epochs
    config.num_epochs = epochs
    (trained_model, history) = train_model(model, train_loader, val_loader, config, class_weights)
    config.num_epochs = original_epochs
    device = torch.device(config.device)
    trained_model.to(device)
    (metrics, y_true, y_pred, y_prob) = validate(trained_model, val_loader, nn.CrossEntropyLoss(), device)
    return (trained_model, history, metrics)

def create_benchmark_table(results_dict, save_path=None):
    """
    Create benchmark results table.
    
    Args:
        results_dict: Dictionary of model results
        save_path: Path to save CSV
    """
    if not results_dict:
        df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)', 'AUROC (Macro)', 'Parameters'])
        print('No benchmark results yet. Train models first.')
        return df
    data = []
    for (model_name, result) in results_dict.items():
        metrics = result['metrics']
        data.append({'Model': model_name, 'Accuracy': metrics.get('accuracy', 0), 'Precision (Macro)': metrics.get('precision_macro', 0), 'Recall (Macro)': metrics.get('recall_macro', 0), 'F1 (Macro)': metrics.get('f1_macro', 0), 'AUROC (Macro)': metrics.get('auroc_macro', 0), 'Parameters': count_parameters(result['model'])})
    df = pd.DataFrame(data)
    df = df.sort_values('F1 (Macro)', ascending=False)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f'Saved to {save_path}')
    return df


def detect_pqrst(signal_data, fs=100):
    """
    Detect P, Q, R, S, and T wave peaks in an ECG signal.
    
    Args:
        signal_data: 1D array of ECG signal values
        fs: Sampling frequency in Hz (default 100)
        
    Returns:
        Dictionary with wave labels as keys and lists of indices as values
    """
    if torch.is_tensor(signal_data):
        signal_data = signal_data.detach().cpu().numpy()
        
    # 1. R-peaks detection (highest peaks)
    # Using a reasonable height threshold and minimum distance between beats
    peaks, _ = signal.find_peaks(signal_data, 
                                distance=int(fs * 0.6), 
                                height=np.mean(signal_data) + 0.5 * np.std(signal_data))
    
    waves = {'P': [], 'Q': [], 'R': peaks.tolist(), 'S': [], 'T': []}
    
    for r_idx in peaks:
        # Search windows relative to R-peak
        # Q search (local minimum before R)
        q_start = max(0, r_idx - int(0.05 * fs))
        q_window = signal_data[q_start:r_idx]
        if len(q_window) > 0:
            waves['Q'].append(q_start + np.argmin(q_window))
            
        # S search (local minimum after R)
        s_end = min(len(signal_data), r_idx + int(0.05 * fs))
        s_window = signal_data[r_idx:s_end]
        if len(s_window) > 0:
            waves['S'].append(r_idx + np.argmin(s_window))
            
        # P search (local maximum before Q)
        p_start = max(0, r_idx - int(0.25 * fs))
        p_end = max(0, r_idx - int(0.05 * fs))
        p_window = signal_data[p_start:p_end]
        if len(p_window) > 0:
            waves['P'].append(p_start + np.argmax(p_window))
            
        # T search (local maximum after S)
        t_start = min(len(signal_data), r_idx + int(0.12 * fs))
        t_end = min(len(signal_data), r_idx + int(0.45 * fs))
        t_window = signal_data[t_start:t_end]
        if len(t_window) > 0:
            waves['T'].append(t_start + np.argmax(t_window))
            
    return waves

class ECGExplainer:
    """
    SHAP explainer for ECG classification models.
    """

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.background = None
        self.explainer = None

    def fit(self, background_data, n_background=50):
        """
        Fit SHAP explainer with background data.
        
        Args:
            background_data: DataLoader or array of background ECGs
            n_background: Number of background samples
        """
        background_ecgs = []
        count = 0
        if hasattr(background_data, '__iter__'):
            for batch in background_data:
                if isinstance(batch, dict):
                    ecg = batch['ecg']
                else:
                    ecg = batch
                background_ecgs.append(ecg.detach().cpu().numpy() if torch.is_tensor(ecg) else ecg)
                count += ecg.shape[0]
                if count >= n_background:
                    break
        self.background = np.concatenate(background_ecgs, axis=0)[:n_background]
        print(f'Using {self.background.shape[0]} background samples for SHAP')

        def predict_fn(X):
            X_reshaped = X.reshape(X.shape[0], 12, -1)
            X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            return probs
        background_flat = self.background.reshape(self.background.shape[0], -1)
        self.explainer = shap.KernelExplainer(predict_fn, background_flat[:10])
        self.predict_fn = predict_fn
        print('SHAP explainer initialized.')

    def explain_instance(self, ecg, class_names=None, nsamples=100):
        """
        Explain a single ECG instance.
        
        Args:
            ecg: ECG array (12, 1000) or (1, 12, 1000)
            class_names: List of class names
            nsamples: Number of samples for SHAP
        
        Returns:
            SHAP values, prediction info
        """
        if class_names is None:
            class_names = config.class_names
        if ecg.ndim == 2:
            ecg = ecg[np.newaxis, ...]
        ecg_tensor = torch.FloatTensor(ecg).to(self.device)
        with torch.no_grad():
            logits = self.model(ecg_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            conf = probs[0, pred_class].item()
        ecg_flat = ecg.reshape(1, -1)
        shap_values = self.explainer.shap_values(ecg_flat, nsamples=nsamples)
        return {'shap_values': shap_values, 'prediction': class_names[pred_class], 'prediction_idx': pred_class, 'confidence': conf, 'probabilities': probs.detach().cpu().numpy()[0], 'ecg': ecg[0]}

    def plot_explanation(self, explanation, save_path=None):
        """Plot SHAP explanation with R-R highlights and PQRST labels on all leads."""
        import ecg_plot
        from matplotlib.collections import LineCollection
        from scipy.ndimage import gaussian_filter1d
        
        shap_values = explanation['shap_values']
        ecg = explanation['ecg']
        pred_class = explanation['prediction_idx']
        
        if isinstance(shap_values, list): class_shap = shap_values[pred_class][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: class_shap = shap_values[0, :, pred_class]
        else: class_shap = shap_values[0]
            
        shap_reshaped = class_shap.reshape(12, 1000)
        sample_rate = 100
        n = ecg.shape[1]
        q1, q2, q3 = n // 4, n // 2, 3 * n // 4

        # Layout mapping
        ecg_new = np.stack([
            np.concatenate([ecg[0, 0:q1], ecg[3, q1:q2], ecg[6, q2:q3], ecg[9, q3:n]]),
            np.concatenate([ecg[1, 0:q1], ecg[4, q1:q2], ecg[7, q2:q3], ecg[10, q3:n]]),
            np.concatenate([ecg[2, 0:q1], ecg[5, q1:q2], ecg[8, q2:q3], ecg[11, q3:n]]),
            ecg[1, :] # Lead II Ref
        ], axis=0)

        shap_new = np.stack([
            np.concatenate([shap_reshaped[0, 0:q1], shap_reshaped[3, q1:q2], shap_reshaped[6, q2:q3], shap_reshaped[9, q3:n]]),
            np.concatenate([shap_reshaped[1, 0:q1], shap_reshaped[4, q1:q2], shap_reshaped[7, q2:q3], shap_reshaped[10, q3:n]]),
            np.concatenate([shap_reshaped[2, 0:q1], shap_reshaped[5, q1:q2], shap_reshaped[8, q2:q3], shap_reshaped[11, q3:n]]),
            shap_reshaped[1, :]
        ], axis=0)

        plt.clf(); plt.close('all')
        lead_names_new = ['I', 'II', 'III', 'II Ref']
        title = f"SHAP Explanation: {explanation['prediction']} (Confidence: {explanation['confidence']:.2f})"
        ecg_plot.plot(ecg_new, sample_rate=sample_rate, title=title, columns=1, lead_index=lead_names_new)
        
        fig = plt.gcf(); fig.set_size_inches(16, 12); ax = plt.gca()
        lines = ax.get_lines(); ecg_lines = [line for line in lines if len(line.get_xdata()) == n]
        shap_max = max(np.abs(shap_new).max(), 1e-9)
        shap_smoothed = gaussian_filter1d(shap_new, sigma=12, axis=1)

        for i, line in enumerate(ecg_lines):
            line.set_visible(False)
            x = line.get_xdata(); y = line.get_ydata(); offset = np.mean(y - ecg_new[i])
            
            # 1. R-R Interval Highlights (Alternating)
            waves = detect_pqrst(ecg_new[i])
            r_peaks = waves['R']
            for r_idx in range(len(r_peaks) - 1):
                if r_idx % 2 == 0:
                    ax.axvspan(x[r_peaks[r_idx]], x[r_peaks[r_idx+1]], color='lightgray', alpha=0.15, zorder=0)

            # 2. Background Glow
            extent = [x[0], x[-1], offset - 3.0, offset + 3.0]
            ax.imshow(shap_smoothed[i:i+1], aspect='auto', cmap='Spectral_r', extent=extent, alpha=0.35, 
                      vmin=-shap_max, vmax=shap_max, zorder=1, interpolation='bicubic')
            
            # 3. Continuous Signal Line
            pts = np.array([x, y]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='Spectral_r', zorder=3)
            lc.set_array(shap_new[i]); lc.set_linewidth(2.5); lc.set_alpha(0.85); lc.set_clim(-shap_max, shap_max)
            ax.add_collection(lc)
            
            # 4. PQRST Labels on ALL leads
            for w_type, indices in waves.items():
                for idx in indices[:4]: # Label first 4 beats
                    ax.text(x[idx], y[idx] + 0.35, w_type, fontsize=8, ha='center', 
                            color='darkred', fontweight='bold', zorder=10)

        ax.set_yticklabels([]); xlabels = ax.get_xticks()
        ax.set_xticklabels([f"{int(x)}s" if x%1==0 and x<=10 else "" for x in xlabels])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved SHAP explanation to {save_path}")
        ecg_plot.show()
class DeepECGSSL(nn.Module):
    """
    DeepECG-SSL: Self-supervised ECG Transformer model adapted from HeartWise-AI.

    Architecture:
    1. Convolutional feature extraction (4 layers)
    2. Positional encoding
    3. Transformer encoder (configurable layers)
    4. Mean pooling + classification head

    Reference: https://github.com/HeartWise-AI/DeepECG-SSL-finetune
    """

    def __init__(self, num_leads=12, num_classes=7, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        # Convolutional feature extraction (similar to wav2vec2 style)
        # Each layer: (out_channels, kernel_size, stride)
        self.feature_extractor = nn.Sequential(
            ConvBlock1D(num_leads, 256, kernel_size=10, stride=2, padding=4),
            ConvBlock1D(256, 256, kernel_size=8, stride=2, padding=3),
            ConvBlock1D(256, 256, kernel_size=4, stride=2, padding=1),
            ConvBlock1D(256, 256, kernel_size=4, stride=2, padding=1),
        )

        # Layer normalization after feature extraction
        self.layer_norm = nn.LayerNorm(256)

        # Project to embedding dimension if needed
        self.post_extract_proj = nn.Linear(256, embed_dim) if embed_dim != 256 else None

        # Convolutional positional encoding
        self.conv_pos = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=128,
            padding=128 // 2,
            groups=16
        )
        self.conv_pos_activation = nn.GELU()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.final_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def _compute_output_length(self, input_length):
        """Compute sequence length after convolutions."""
        # Each conv layer: output = floor((input - kernel + 2*padding) / stride + 1)
        length = input_length
        for _ in range(4):  # 4 conv layers with stride 2
            length = (length - 1) // 2 + 1
        return length

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, num_leads, seq_len)

        Returns:
            logits: Output tensor of shape (batch, num_classes)
        """
        # Feature extraction
        features = self.feature_extractor(x)  # (B, 256, T')
        features = features.transpose(1, 2)  # (B, T', 256)
        features = self.layer_norm(features)

        # Project to embedding dimension
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)  # (B, T', embed_dim)

        # Add positional encoding
        # Conv1d expects (B, C, T)
        pos_conv_input = features.transpose(1, 2)  # (B, embed_dim, T')
        pos_encoding = self.conv_pos(pos_conv_input)  # (B, embed_dim, T')
        pos_encoding = self.conv_pos_activation(pos_encoding)
        pos_encoding = pos_encoding.transpose(1, 2)  # (B, T', embed_dim)

        # Add positional encoding to features
        x = features + pos_encoding

        # Transformer encoding
        x = self.transformer(x)  # (B, T', embed_dim)

        # Mean pooling over time dimension (ignoring zero-padded positions)
        # Simple mean pooling
        x = x.mean(dim=1)  # (B, embed_dim)

        # Classification
        x = self.final_dropout(x)
        logits = self.classifier(x)  # (B, num_classes)

        return logits


class HybridNoGate(nn.Module):
    """Hybrid model without gated fusion (simple concatenation)."""

    def __init__(self, num_leads=12, num_classes=7):
        super().__init__()
        self.cnn = nn.Sequential(ConvBlock1D(num_leads, 64, 7, stride=2, padding=3), nn.MaxPool1d(2), ConvBlock1D(64, 128, 5, stride=2, padding=2), ConvBlock1D(128, 256, 5, stride=2, padding=2), nn.AdaptiveAvgPool1d(1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 2)
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        cnn_feat = self.cnn(x).squeeze(-1)
        patches = x.permute(0, 2, 1)
        trans_feat = self.transformer(patches).mean(dim=1)
        combined = torch.cat([cnn_feat, trans_feat], dim=1)
        return self.classifier(combined)

class CNNOnly(nn.Module):
    """CNN-only model (no transformer)."""

    def __init__(self, num_leads=12, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(ConvBlock1D(num_leads, 64, 7, stride=2, padding=3), nn.MaxPool1d(2), ConvBlock1D(64, 128, 5, stride=2, padding=2), ConvBlock1D(128, 256, 5, stride=2, padding=2), ConvBlock1D(256, 512, 3, stride=1, padding=1), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

def run_ablation_study(ablations, train_loader, val_loader, config, class_weights, epochs=15):
    """
    Run ablation experiments.
    
    Args:
        ablations: Dict of {name: model_class}
    
    Returns:
        DataFrame of ablation results
    """
    results = []
    for (name, model_class) in ablations.items():
        print(f"\n{'=' * 50}")
        print(f'Ablation: {name}')
        print(f"{'=' * 50}")
        model = model_class()
        device = torch.device(config.device)
        model = model.to(device)
        (trained_model, history) = train_model(model, train_loader, val_loader, config, class_weights)
        (metrics, _, _, _) = validate(trained_model, val_loader, nn.CrossEntropyLoss(), device)
        results.append({'Model': name, 'Accuracy': metrics.get('accuracy', 0), 'F1_Macro': metrics.get('f1_macro', 0), 'AUROC': metrics.get('auroc_macro', 0), 'Parameters': count_parameters(trained_model), 'Best_Val_Loss': min(history['val_loss'])})
    return pd.DataFrame(results)

def plot_ablation_results(ablation_df, save_path=None):
    """Plot ablation study results."""
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(ablation_df))
    width = 0.6
    axes[0].bar(x, ablation_df['Accuracy'], width, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Model Variant')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ablation_df['Model'], rotation=15, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[1].bar(x, ablation_df['F1_Macro'], width, color='coral', alpha=0.7)
    axes[1].set_xlabel('Model Variant')
    axes[1].set_ylabel('F1 (Macro)')
    axes[1].set_title('F1 Score Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ablation_df['Model'], rotation=15, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[2].bar(x, ablation_df['Parameters'] / 1000000.0, width, color='seagreen', alpha=0.7)
    axes[2].set_xlabel('Model Variant')
    axes[2].set_ylabel('Parameters (M)')
    axes[2].set_title('Parameter Count')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ablation_df['Model'], rotation=15, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()

def generate_test_predictions(model, test_loader, device, threshold=0.7, output_path=None, class_names=None):
    """
    Generate predictions for test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        threshold: Confidence threshold for flagging
        output_path: Path to save predictions CSV
        class_names: List of class names
    
    Returns:
        DataFrame of predictions
    """
    if class_names is None:
        class_names = config.class_names
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating predictions'):
            ecg = batch['ecg'].to(device)
            filename = batch.get('filename', ['unknown'])[0]
            outputs = model(ecg)
            probs = F.softmax(outputs, dim=1)
            (max_prob, pred) = probs.max(dim=1)
            (top3_probs, top3_indices) = probs.topk(3, dim=1)
            pred_data = {'filename': filename, 'predicted_class': class_names[pred.item()], 'confidence': max_prob.item(), 'low_confidence_flag': max_prob.item() < threshold}
            for (i, cls) in enumerate(class_names):
                pred_data[f'prob_{cls}'] = probs[0, i].item()
            pred_data['top2_class'] = class_names[top3_indices[0, 1].item()]
            pred_data['top2_confidence'] = top3_probs[0, 1].item()
            pred_data['top3_class'] = class_names[top3_indices[0, 2].item()]
            pred_data['top3_confidence'] = top3_probs[0, 2].item()
            predictions.append(pred_data)
    df = pd.DataFrame(predictions)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f'Predictions saved to {output_path}')
    return df

def generate_validation_predictions(model, val_loader, device, threshold=0.7, output_path=None, class_names=None):
    """
    Generate predictions for validation set with true labels.
    
    Returns:
        DataFrame with predictions and true labels
    """
    if class_names is None:
        class_names = config.class_names
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Generating validation predictions'):
            ecg = batch['ecg'].to(device)
            filename = batch.get('filename', ['unknown'])[0]
            true_label = batch['label'].squeeze().item()
            true_class = class_names[true_label]
            outputs = model(ecg)
            probs = F.softmax(outputs, dim=1)
            (max_prob, pred) = probs.max(dim=1)
            predictions.append({'filename': filename, 'true_class': true_class, 'predicted_class': class_names[pred.item()], 'confidence': max_prob.item(), 'correct': pred.item() == true_label, 'low_confidence_flag': max_prob.item() < threshold})
    df = pd.DataFrame(predictions)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f'Validation predictions saved to {output_path}')
    return df

def estimate_carbon_footprint(training_time_hours, hardware_type='GPU', num_gpus=1, gpu_type='Unknown'):
    """
    Estimate carbon footprint of training.
    
    Based on: https://mlco2.github.io/impact#compute
    
    Approximate emissions:
    - US GPU: ~0.05 kg CO2e per GPU-hour
    - US CPU: ~0.02 kg CO2e per CPU-hour
    - Europe GPU: ~0.02 kg CO2e per GPU-hour
    """
    emission_factors = {('US', 'GPU'): 0.05, ('US', 'CPU'): 0.02, ('Europe', 'GPU'): 0.02, ('Europe', 'CPU'): 0.01}
    region = 'US'
    factor = emission_factors.get((region, hardware_type), 0.03)
    total_emissions = training_time_hours * num_gpus * factor
    return {'training_time_hours': training_time_hours, 'hardware_type': hardware_type, 'num_gpus': num_gpus, 'gpu_type': gpu_type, 'estimated_emissions_kg_co2e': total_emissions, 'emissions_equivalent_miles_driven': total_emissions * 2.5, 'region_assumed': region}

def generate_summary_report(config, output_path):
    """Generate a comprehensive summary report."""
    report = f"\n# ECG Classification Project Summary Report\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n## Project Overview\n\n### Task\nMulti-class 12-lead ECG arrhythmia classification using hybrid CNN-Transformer architecture.\n\n### Target Classes\n{', '.join(config.class_names)}\n\n### Dataset Summary\n| Dataset | Use | Records | Source |\n|---------|-----|---------|--------|\n| PTB-XL | Training | 21,837 | PhysioNet |\n| Challenge 2020 | External Validation | 43,301 | PhysioNet |\n| Tuning Set | Tuning | 6 | Course Assignment |\n| Test | Final Evaluation | 6 | Course Assignment |\n\n## Model Architecture\n\n### Final Model: Hybrid CNN-Transformer with ResNet Backbone\n- **Input:** 12-lead ECG (1000 time steps at 100 Hz)\n- **Backbone:** ResNet-style CNN for local morphology\n- **Context:** Transformer encoder for long-range dependencies\n- **Fusion:** Gated mechanism combining local and global features\n- **Output:** 7-class logits with temperature scaling\n\n## Evaluation Metrics\n\n### Metrics Computed\n- Accuracy\n- Precision (Macro, Weighted, Per-class)\n- Recall/Sensitivity (Macro, Weighted, Per-class)\n- F1 Score (Macro, Weighted, Per-class)\n- AUROC (Macro, Per-class)\n- AUPRC (Macro, Per-class)\n- Confidence calibration metrics\n\n## Explainability\n\n- **Method:** SHAP (Kernel SHAP for compatibility)\n- **Output:** Lead-wise contribution scores\n- **Visualization:** Heatmaps overlaid on ECG signals\n\n## Low-Confidence Detection\n\n- **Method:** Maximum softmax probability thresholding\n- **Default Threshold:** {config.confidence_threshold}\n- **Output:** Flagged cases for manual review\n\n## Ethical Considerations\n\n### Australia's AI Ethics Principles\nAll 8 principles addressed:\n1. ✅ Human, social and environmental wellbeing\n2. ✅ Human-centred values\n3. ✅ Fairness (with noted limitations)\n4. ✅ Privacy protection and security\n5. ✅ Reliability and safety\n6. ✅ Transparency and explainability\n7. ✅ Contestability\n8. ✅ Accountability\n\n### UN SDG Alignment\n- **SDG 3:** Good Health and Well-Being (primary)\n- **SDG 9:** Industry, Innovation and Infrastructure (secondary)\n- **SDG 10:** Reduced Inequalities (potential, requires validation)\n\n### Carbon Footprint\n- Emissions tracked using codecarbon\n- Early stopping to minimize training time\n- Estimated emissions saved to carbon_footprint_estimate.md\n\n## Output Files\n\n### Data Files\n- `dataset_summary.csv` - Dataset information\n- `class_distribution_validation.csv` - Validation class distribution\n- `signal_statistics.csv` - Signal statistics\n\n### Model Files\n- `model_info.json` - Model architecture details\n- `model_card.md` - Model documentation\n\n### Results Files\n- `benchmark_results.csv` - Model comparison\n- `qualitative_comparison.csv` - Qualitative model comparison\n- `ablation_results.csv` - Ablation study results\n\n### Prediction Files\n- `test_predictions.csv` - Test set predictions\n- `low_confidence_cases.csv` - Low-confidence predictions\n\n### Ethics and Documentation\n- `ethics_checklist.md` - AI ethics compliance\n- `carbon_footprint_estimate.md` - Environmental impact\n- `sdg_alignment.md` - UN SDG alignment\n- `deployment_considerations.json` - Deployment analysis\n\n### Figures\n- `sample_ecg_validation*.png` - Sample ECG visualizations\n- `class_distribution_validation.png` - Class distribution plot\n- `lead_correlation.png` - Lead correlation matrix\n- `training_history.png` - Training curves\n- `confusion_matrix.png` - Confusion matrix\n- `roc_curves.png` - ROC curves\n- `shap_example_*.png` - SHAP explanations\n\n## Configuration\n\n- **Random Seed:** {config.random_seed}\n- **Batch Size:** {config.batch_size}\n- **Learning Rate:** {config.learning_rate}\n- **Early Stopping Patience:** {config.early_stopping_patience}\n- **Device:** {config.device}\n\n---\n\n*Report generated by COMP6011 Task 3 Notebook*\n"
    with open(output_path, 'w') as f:
        f.write(report)

def list_output_files(output_dir):
    """List all files in the output directory."""
    files = sorted(output_dir.glob('*'))
    categories = {'Data': ['csv', 'json'], 'Documentation': ['md', 'txt'], 'Figures': ['png', 'jpg', 'svg']}
    print('\n' + '=' * 60)
    print('OUTPUT FILES GENERATED')
    print('=' * 60)
    for (category, extensions) in categories.items():
        category_files = [f for f in files if f.suffix[1:] in extensions]
        if category_files:
            print(f'\n{category}:')
            for f in category_files:
                size_kb = f.stat().st_size / 1024
                print(f'  - {f.name} ({size_kb:.1f} KB)')
    print('\n' + '=' * 60)
class ECGGradCAM:
    """Grad-CAM for ECG models."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        output[0, class_idx].backward()
        
        # Pool the gradients across the temporal dimension
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)
        
        # Upsample to original size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                            size=(input_tensor.size(2),), 
                            mode='linear', align_corners=False).squeeze()
        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
