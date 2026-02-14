import argparse
import os
from pathlib import Path
import urllib.request
import pandas as pd
import torch
from esm import FastaBatchedDataset, pretrained
from tqdm.auto import tqdm

from configs.configs_model import ESM_CONFIG

def download_with_progress(url, filepath):
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    urllib.request.urlretrieve(url, filepath, progress_hook)
    print()  # 换行

def download_esm_model(model_name='esm2-3b', model_path = None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_name=ESM_CONFIG.get(model_name)
    if model_name is None:
        raise ValueError(f"Model {model_name} not found in ESM_CONFIG. Available models: {list(ESM_CONFIG.keys())}")
    if model_path is None:
        model_path=ESM_CONFIG.get(model_name).get('model_path',None)
    url=model_name['model_url']

    print(f"开始下载主模型到: {model_path}")
    download_with_progress(url, model_path)
    print("下载完成!")

    contact_model_url = model_name.get('contact_model_url',None)
    if contact_model_url is not None:
        contact_model_path = os.path.join(os.path.dirname(model_path), os.path.basename(contact_model_url))
        print(f"开始下载接触图模型到: {contact_model_path}")
        download_with_progress(contact_model_url, contact_model_path)
        print("下载完成!")

def _load_esm2_model(model_path):
    if os.path.exists(model_path):
        model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
    else:
        download_esm_model(model_name=os.path.splitext(os.path.basename(model_path))[0],model_path=model_path)
        model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
        # model, alphabet = pretrained.load_model_and_alphabet(
        #     os.path.splitext(os.path.basename(model_path))[0]
        # )
    return model, alphabet

def load_model_and_alphabet_local(model_location, device_type="cpu", device=None):
    """
    从本地路径加载模型。回归权重需要位于同一位置
    
    Args:
        model_location: 模型文件路径
        device_type: 设备类型，如'cpu'或'cuda'
        device: 特定的设备，如'cuda:0'、'cuda:1'等，优先级高于device_type
        
    Returns:
        加载的模型和字母表
    """
    model_location = Path(model_location)
    map_location = device if device is not None else device_type
    model_data = torch.load(str(model_location), map_location=map_location, weights_only=False)
    model_name = model_location.stem
    if pretrained._has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location=map_location, weights_only=False)
    else:
        regression_data = None
    return pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)

def _load_esm2_model(model_path, device_type="cpu", device=None):
    if os.path.exists(model_path):
        model, alphabet = load_model_and_alphabet_local(model_path, device_type=device_type, device=device)
    else:
        try:
            model, alphabet = pretrained.load_model_and_alphabet(
                os.path.splitext(os.path.basename(model_path))[0]
            )
        except Exception as e:
            print(f"Error loading ESM model: {e},\n Trying to download model by wget...")
            download_esm_model(model_name=os.path.splitext(os.path.basename(model_path))[0],model_path=model_path)
            model, alphabet = load_model_and_alphabet_local(model_path, device_type=device_type, device=device)

    return model, alphabet


def load_esm_model(model_name, local_esm_model_dir="/root/.cache/torch/hub/checkpoints/", device_type="cpu", device=None):
    local_model_path = os.path.join(local_esm_model_dir, ESM_CONFIG[model_name]["model_path"])
    print("Try to load ESM language model from ", local_model_path)

    if model_name.startswith("esm2"):
        model, alphabet = _load_esm2_model(local_model_path, device_type=device_type, device=device)
    
    model.eval()
    
    if device is not None:
        model = model.to(device)
        print(f"ESM model moved to device: {device}")
    elif device_type == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"ESM model moved to CUDA (GPU)")
    else:
        print(f"ESM model using CPU")

    return model, alphabet


def _check_files_exist(save_dir, labels):
    return all(
        [os.path.exists(os.path.join(save_dir, label + ".pt")) for label in labels]
    )


def compute_ESM_embeddings(
    model_name,
    model,
    alphabet,
    labels,
    sequences,
    save_dir,
    toks_per_batch=4096,
    truncation_seq_length=1022,
    device=None,
):
    assert model_name.startswith("esm2")
    embeddings = compute_esm2_embeddings(
        model,
        alphabet,
        labels,
        sequences,
        save_dir,
        toks_per_batch,
        truncation_seq_length,
        device=device,
    )
    return embeddings


# Adapt from Corso, Gabriele, et al. "Diffdock: Diffusion steps, twists, and turns for molecular docking."
# URL: https://github.com/gcorso/DiffDock/blob/main/utils/inference_utils.py
def compute_esm2_embeddings(
    model,
    alphabet,
    labels,
    sequences,
    save_dir,
    toks_per_batch=4096,
    truncation_seq_length=1022,
    device=None,
):
    # 自动检测设备
    if device is None:
        device = next(model.parameters()).device
    
    print(f"Computing ESM embeddings on device: {device}")
    
    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )
    repr_layer = model.num_layers
    embeddings = {}
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if _check_files_exist(save_dir, labels):
                continue
            
            toks = toks.to(device=device, non_blocking=True)
            
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            representation = out["representations"][repr_layer].to(device="cpu")
            
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representation[i, 1 : truncate_len + 1].clone()
                save_path = os.path.join(save_dir, label + ".pt")
                torch.save(embeddings[label], save_path)
    
    return embeddings


def pdb_sequences_iterator(
    input_path="pdb_seq.csv",
    save_path="pdb_labels_seqs.csv",
    start_id=0,
    end_id=-1,
    protein_only=True,
    force_regenerate=False,
):
    """
    生成或读取序列标签文件
    
    Args:
        input_path: 输入的 pdb_seq.csv 文件路径
        save_path: 输出的 pdb_labels_seqs.csv 文件路径
        start_id: 起始序列ID
        end_id: 结束序列ID (-1表示全部)
        protein_only: 是否只保留蛋白质序列 (True=只保留protein, False=保留所有类型)
        force_regenerate: 是否强制重新生成标签文件
    """
    need_regenerate = False
    
    if os.path.exists(save_path) and not force_regenerate:
        # 检查标签文件是否完整
        df_seq_existing = pd.read_csv(save_path)
        df_input = pd.read_csv(input_path)
        
        # 根据 protein_only 过滤输入数据
        if protein_only:
            df_input_filtered = df_input[df_input["mol_type"] == "protein"]
        else:
            df_input_filtered = df_input
        
        # 计算应该有多少唯一序列
        df_input_filtered["pdb_entity_id"] = df_input_filtered["pdb_id"] + "_" + df_input_filtered["entity_id"].astype(str)
        expected_unique_seqs = df_input_filtered.groupby("seq").size().shape[0]
        actual_unique_seqs = len(df_seq_existing)
        
        # 如果数量不匹配，需要重新生成
        if actual_unique_seqs < expected_unique_seqs:
            print(f"⚠️  标签文件不完整: 期望 {expected_unique_seqs} 个唯一序列，实际只有 {actual_unique_seqs} 个")
            print(f"   将重新生成标签文件...")
            need_regenerate = True
        else:
            df_seq = df_seq_existing
    else:
        need_regenerate = True
    
    if need_regenerate:
        df = pd.read_csv(input_path)
        # 根据 protein_only 参数决定是否过滤
        if protein_only:
            df = df[df["mol_type"] == "protein"]
            print(f"✓ 过滤后保留 {len(df)} 行蛋白质序列")
        else:
            print(f"✓ 保留所有类型的序列: {len(df)} 行")
            mol_type_counts = df["mol_type"].value_counts()
            for mol_type, count in mol_type_counts.items():
                print(f"  {mol_type}: {count}")
        
        # Sequence name
        df["pdb_entity_id"] = df["pdb_id"] + "_" + df["entity_id"].astype(str)
        # Group by 'seq'
        df_seq = (
            df.groupby("seq")["pdb_entity_id"]
            .apply(lambda x: ",".join(x))
            .reset_index()
        )
        # Use the first pdb_entity_id as the label
        df_seq["seq_label"] = df_seq["pdb_entity_id"].apply(lambda x: x.split(",")[0])
        assert df_seq["seq_label"].nunique() == len(df_seq)
        # Get a part id
        df_seq["part_id"] = df_seq["pdb_entity_id"].apply(lambda x: x[1:3])
        df_seq.to_csv(save_path)
        print(f"✓ 已生成标签文件: {save_path}，包含 {len(df_seq)} 个唯一序列")

    if end_id == -1:
        end_id = len(df_seq)
    df_seq = df_seq[start_id:end_id]

    part_counts = dict(df_seq["part_id"].value_counts())
    for part_id, count in part_counts.items():
        df_part = df_seq[df_seq["part_id"] == part_id]
        print(f"Part {part_id}: {len(df_part)} sequences.")
        yield part_id, df_part["seq_label"].tolist(), df_part["seq"].tolist()


def process_pdb_dataset(
    model_name, root_save_dir, pdb_seq_path, pdb_seq_label_path, start_id=0, end_id=-1,local_esm_model_dir="/root/.cache/torch/hub/checkpoints/",
    device_type="cpu", device=None, protein_only=True, force_regenerate=False
):

    model, alphabet = load_esm_model(model_name,local_esm_model_dir=local_esm_model_dir, device_type=device_type, device=device)
    seq_iterator = pdb_sequences_iterator(
        pdb_seq_path, pdb_seq_label_path, start_id, end_id, 
        protein_only=protein_only, force_regenerate=force_regenerate
    )
    error_parts = []
    for part_id, labels, sequences in seq_iterator:
        save_dir = os.path.join(root_save_dir, f"{part_id}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[{part_id}] Generating ESM language model embeddings")
        lm_embeddings = compute_ESM_embeddings(
            model_name,
            model,
            alphabet,
            labels,
            sequences,
            save_dir,
            truncation_seq_length=4094,
            toks_per_batch=16384,
            device=device,  # 传递 device 参数
        )
        print(f"[{part_id}] Processed {len(lm_embeddings)} sequences in total. Done!")

    print("Error parts: ", error_parts)


def main():
    parser = argparse.ArgumentParser(description="Compute ESM embeddings for protein sequences")
    parser.add_argument("--model_name", type=str, choices=list(ESM_CONFIG.keys()), required=True,
                       help="ESM model name to use")
    parser.add_argument("--start_id", type=int, default=0,
                       help="Start sequence ID (default: 0)")
    parser.add_argument("--end_id", type=int, default=-1,
                       help="End sequence ID, -1 for all (default: -1)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="GPU ID to use (e.g., 0, 1, 2, 3). Overrides --device if specified.")
    home_dir = os.path.expanduser("~")
    parser.add_argument("--local_esm_model_dir", type=str, default=f"{home_dir}/.cache/torch/hub/checkpoints/",
                       help="Directory containing ESM model files")
    parser.add_argument("--pdb_seq_path", type=str, default="./data/pdb_seqs/pdb_seq.csv",
                       help="Path to PDB sequence CSV file")
    parser.add_argument("--pdb_seq_label_path", type=str, default="./data/pdb_seqs/pdb_labels_seqs.csv",
                       help="Path to PDB sequence label CSV file")
    parser.add_argument("--esm_save_dir", type=str, default=None,
                       help="Directory to save ESM embeddings (default: from ESM_CONFIG)")
    parser.add_argument("--protein_only", action="store_true", default=True,
                       help="Only process protein sequences (default: True)")
    parser.add_argument("--include_all_types", action="store_true", default=False,
                       help="Include all molecule types (protein, na, other). Overrides --protein_only")
    parser.add_argument("--force_regenerate", action="store_true", default=False,
                       help="Force regenerate pdb_labels_seq.csv even if it exists")
    
    args = parser.parse_args()
    
    # 处理 protein_only 参数
    protein_only = args.protein_only and not args.include_all_types
    
    # ============================================================================
    # Device Selection
    # ============================================================================
    print("=" * 80)
    print("ESM Embedding Computation")
    print("=" * 80)
    
    # 检测可用的 GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA is available!")
        print(f"  GPU count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("✗ CUDA is not available, using CPU")
    
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"→ Using specified GPU {args.gpu_id}: cuda:{args.gpu_id}")
    elif args.device is not None:
        device = torch.device(args.device)
        print(f"→ Using specified device: {args.device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"→ Auto-detected device: {device}")
    
    print("=" * 80)
    
    # ============================================================================
    # Configuration
    # ============================================================================
    print(f"Model: {args.model_name}")
    print(f"Local ESM model directory: {args.local_esm_model_dir}")
    
    if args.esm_save_dir is None:
        args.esm_save_dir = ESM_CONFIG[args.model_name]["emb_dir"]
    print(f"Save directory: {args.esm_save_dir}")
    
    if not os.path.exists(args.esm_save_dir):
        print(f"Creating save directory: {args.esm_save_dir}")
        os.makedirs(args.esm_save_dir)
    
    print("=" * 80)
    
    # ============================================================================
    # Process Dataset
    # ============================================================================
    print(f"Protein only: {protein_only}")
    if args.force_regenerate:
        print("Force regenerate: True (will regenerate pdb_labels_seq.csv)")
    print("=" * 80)
    
    process_pdb_dataset(
        args.model_name,
        args.esm_save_dir,
        args.pdb_seq_path,
        args.pdb_seq_label_path,
        args.start_id,
        args.end_id,
        local_esm_model_dir=args.local_esm_model_dir,
        device=device,
        protein_only=protein_only,
        force_regenerate=args.force_regenerate
    )
    
    print("=" * 80)
    print("✓ All sequences processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

