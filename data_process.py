import os
import lmdb
import pandas as pd
import torch
import esm
import hashlib
from igfold import IgFoldRunner
from tqdm import tqdm
import pickle
import numpy as np


BASE_PATH = '/home/h666/research/PACNET'
RAW_DATA_PATH = os.path.join(BASE_PATH, "datasets/process_data/COVID-19/Cov_with_target_split.csv")
PROCESSED_PATH = os.path.join(BASE_PATH, "datasets/processed/COVID-19")
ANTIGEN_ESM_PATH = os.path.join(PROCESSED_PATH, "antigen_esm")
ANTIBODY_STRUCTURE_PATH = os.path.join(PROCESSED_PATH, "antibody_igfoldC")
METADATA_PATH = os.path.join(PROCESSED_PATH, "metadata.csv")

os.makedirs(ANTIGEN_ESM_PATH, exist_ok=True)
os.makedirs(ANTIBODY_STRUCTURE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

def get_safe_filename(sequence, extension=".pt"):

    if len(sequence) > 20:
        prefix = sequence[:10]
        suffix_hash = hashlib.md5(sequence.encode()).hexdigest()[:10]
        return f"{prefix}_{suffix_hash}{extension}"
    return f"{sequence}{extension}"

def process(fold=None, train_idx=None, val_idx=None, output_suffix="", data_path=None):

    print("Loading raw data...")
    if data_path is not None:
        df = pd.read_csv(data_path)  
    else:
        df = pd.read_csv(RAW_DATA_PATH)
    required_cols = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Total samples after cleaning: {len(df)}")

    if train_idx is not None and val_idx is not None:
        if data_path is not None:
            print("Warning: data_path provided, ignoring train_idx/val_idx")
            process_idx = np.arange(len(df)) 
        else:
            process_idx = np.concatenate([train_idx, val_idx])
        df = df.iloc[process_idx].reset_index(drop=True)
    elif fold is not None:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(df))
        train_idx, val_idx = splits[fold]
        process_idx = np.concatenate([train_idx, val_idx])
        df = df.iloc[process_idx].reset_index(drop=True)
        print(f"Processing Fold {fold}: {len(df)} samples")

    print("Initializing models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    antigen_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    antigen_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    igfold = IgFoldRunner()
    if hasattr(igfold, 'model'):
        igfold.model = igfold.model.to(device)

    if output_suffix:
        PROCESSED_PATH_FOLD = os.path.join(BASE_PATH, f"datasets/processed/COVID{output_suffix}")
    else:
        PROCESSED_PATH_FOLD = PROCESSED_PATH

    ANTIGEN_ESM_PATH_FOLD = os.path.join(PROCESSED_PATH_FOLD, "antigen_esm")
    ANTIBODY_STRUCTURE_PATH_FOLD = os.path.join(PROCESSED_PATH_FOLD, "antibody_igfold")
    METADATA_PATH_FOLD = os.path.join(PROCESSED_PATH_FOLD, f"metadata{output_suffix}.csv")

    os.makedirs(ANTIGEN_ESM_PATH_FOLD, exist_ok=True)
    os.makedirs(ANTIBODY_STRUCTURE_PATH_FOLD, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH_FOLD), exist_ok=True)

    metadata = []
    success_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            antigen_seq = row['Antigen Sequence']
            antigen_esm_file = get_safe_filename(antigen_seq, ".pt")
            antigen_esm_path = os.path.join(ANTIGEN_ESM_PATH_FOLD, antigen_esm_file)

            if not os.path.exists(antigen_esm_path):
                antigen_batch = [("antigen", antigen_seq)]  
                _, _, antigen_tensor = batch_converter(antigen_batch)
                antigen_tensor = antigen_tensor.to(device)
                with torch.no_grad():
                    results = antigen_model(antigen_tensor, repr_layers=[33])
                    embedding = results["representations"][33].squeeze(0).cpu()
                torch.save(embedding, antigen_esm_path)
                del antigen_tensor, results, embedding
                torch.cuda.empty_cache()

            antibody_seq = row['H-FR1'] + row['H-CDR1'] + row['H-FR2'] + row['H-CDR2'] + row['H-FR3'] + row['H-CDR3'] + row['H-FR4']
            antibody_structure_file = get_safe_filename(antibody_seq, ".pt")
            antibody_structure_path = os.path.join(ANTIBODY_STRUCTURE_PATH_FOLD, antibody_structure_file)

            if not os.path.exists(antibody_structure_path):
                sequences = {"H": antibody_seq}
                with torch.no_grad():
                    emb = igfold.embed(sequences=sequences)
                    structure_emb = emb.structure_embs.detach().cpu()
                torch.save(structure_emb, antibody_structure_path)
                del emb, structure_emb
                torch.cuda.empty_cache()

            metadata.append({
                'index': idx,
                'antigen_esm_path': os.path.relpath(antigen_esm_path, BASE_PATH),
                'antibody_structure_path': os.path.relpath(antibody_structure_path, BASE_PATH),
                'antigen_sequence': antigen_seq,
                'antibody_sequence': antibody_seq
            })
            success_count += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(METADATA_PATH_FOLD, index=False)
    print(f"Metadata saved to {METADATA_PATH_FOLD}")
    print(f"Success: {success_count}/{len(df)} ({success_count/len(df)*100:.2f}%)")

    return METADATA_PATH_FOLD, PROCESSED_PATH_FOLD


if __name__ == "__main__":
    main()
    #create_structure_lmdb()