import os
import pandas as pd
import sys
from pathlib import Path
import multiprocessing
from tqdm import tqdm
from Bio.PDB import MMCIFParser, MMCIFIO
from Bio.PDB.PDBExceptions import PDBConstructionException

# 添加crawlers模块的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crawlers import download_pdb, download_pdb_bulk

def extract_pdb_info(pdb_chain_string):
    """
    从PDB字符串中提取PDB ID和链ID
    例如: '1l2wJ' -> ('1l2w', 'J')
    """
    if len(pdb_chain_string) >= 5:
        pdb_id = pdb_chain_string[:4].lower()
        chain_id = pdb_chain_string[4:]
        return pdb_id, chain_id
    return None, None

def process_atlas_csv(csv_path):
    """
    处理ATLAS.csv文件，添加PDB ID和链ID列
    """
    print("正在读取ATLAS.csv文件...")
    df = pd.read_csv(csv_path)
    
    # 提取PDB ID和链ID
    pdb_info = df['PDB'].apply(extract_pdb_info)
    df['PDB_ID'] = [info[0] for info in pdb_info]
    df['Chain_ID'] = [info[1] for info in pdb_info]
    
    # 保存更新后的CSV文件
    df.to_csv(csv_path, index=False)
    print(f"已更新CSV文件，添加了PDB_ID和Chain_ID列")
    
    return df

def download_cif_files(df, save_dir):
    """
    批量下载CIF文件
    """
    print("开始下载CIF文件...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    download_params = []
    for _, row in df.iterrows():
        pdb_id = row['PDB_ID']
        if pdb_id:  # 只处理有效的PDB ID
            download_params.append((pdb_id, 'atlas', save_dir, ['-c']))
    
    max_workers = min(4, os.cpu_count())  # 限制并发数避免过载
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(download_pdb, download_params),
            total=len(download_params),
            desc="下载CIF文件"
        ))
    
    successful = [r[0] for r in results if r[1]]
    failed = [r[0] for r in results if not r[1]]
    
    print(f"成功下载: {len(successful)} 个CIF文件")
    print(f"下载失败: {len(failed)} 个CIF文件")
    
    if failed:
        print(f"失败的PDB ID: {failed}")
    
    return successful, failed

def extract_chain_from_cif(cif_file_path, chain_id, output_path):
    """
    使用Biopython从CIF文件中提取特定链的结构信息
    """
    try:
        # 创建CIF解析器
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_file_path)
        
        # 定义链选择器
        class ChainSelector:
            def __init__(self, target_chain_id):
                self.target_chain_id = target_chain_id
            
            def accept_chain(self, chain):
                return chain.id == self.target_chain_id
        
        # 使用MMCIFIO保存特定链
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_path, select=ChainSelector(chain_id))
        
        return True
        
    except PDBConstructionException as e:
        print(f"解析CIF文件时出错: {e}")
        return False
    except Exception as e:
        print(f"提取链 {chain_id} 时出错: {e}")
        return False 

# def extract_chain_from_cif(cif_file_path, chain_id, output_path):
#     """
#     从CIF文件中提取特定链的结构信息
#     """
#     try:
#         with open(cif_file_path, 'r') as f:
#             lines = f.readlines()
        
#         # 找到需要提取的链
#         atom_lines = []
#         in_atom_section = False
        
#         for line in lines:
#             line = line.strip()
            
#             # 检查是否在atom_site部分
#             if line.startswith('_atom_site.'):
#                 in_atom_section = True
#                 continue
#             elif line.startswith('_') and not line.startswith('_atom_site.'):
#                 in_atom_section = False
#                 continue
            
#             # 如果在atom_site部分且包含目标链
#             if in_atom_section and line and not line.startswith('#'):
#                 parts = line.split()
#                 if len(parts) >= 4:  # 确保有足够的列
#                     # 假设链ID在第4列（索引3），根据实际CIF格式调整
#                     if len(parts) > 3 and parts[3] == chain_id:
#                         atom_lines.append(line)
        
#         # 写入新的CIF文件
#         with open(output_path, 'w') as f:
#             # 写入CIF头部信息
#             f.write("data_" + os.path.basename(output_path).replace('.cif', '') + "\n")
#             f.write("#\n")
#             f.write("# 从原始CIF文件中提取的链 " + chain_id + "\n")
#             f.write("#\n")
#             f.write("_entry.id " + os.path.basename(output_path).replace('.cif', '') + "\n")
#             f.write("#\n")
#             f.write("_atom_site.group_PDB\n")
#             f.write("_atom_site.id\n")
#             f.write("_atom_site.type_symbol\n")
#             f.write("_atom_site.label_atom_id\n")
#             f.write("_atom_site.label_alt_id\n")
#             f.write("_atom_site.label_comp_id\n")
#             f.write("_atom_site.label_asym_id\n")
#             f.write("_atom_site.label_entity_id\n")
#             f.write("_atom_site.label_seq_id\n")
#             f.write("_atom_site.pdbx_PDB_ins_code\n")
#             f.write("_atom_site.Cartn_x\n")
#             f.write("_atom_site.Cartn_y\n")
#             f.write("_atom_site.Cartn_z\n")
#             f.write("_atom_site.occupancy\n")
#             f.write("_atom_site.B_iso_or_equiv\n")
#             f.write("_atom_site.pdbx_formal_charge\n")
#             f.write("_atom_site.auth_seq_id\n")
#             f.write("_atom_site.auth_comp_id\n")
#             f.write("_atom_site.auth_asym_id\n")
#             f.write("_atom_site.auth_atom_id\n")
#             f.write("_atom_site.pdbx_PDB_model_num\n")
#             f.write("#\n")
            
#             # 写入原子数据
#             for line in atom_lines:
#                 f.write(line + "\n")
        
#         return True
        
#     except Exception as e:
#         print(f"提取链 {chain_id} 时出错: {e}")
#         return False

def process_chain_extraction(df, cif_dir, output_dir):
    """
    处理链提取
    """
    print("开始提取特定链的结构...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    successful_extractions = []
    failed_extractions = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="提取链结构"):
        pdb_id = row['PDB_ID']
        chain_id = row['Chain_ID']
        
        if not pdb_id or not chain_id:
            continue
        
        # 查找对应的CIF文件
        cif_file = os.path.join(cif_dir, 'atlas', f"{pdb_id}.cif")
        
        if not os.path.exists(cif_file):
            print(f"未找到CIF文件: {cif_file}")
            failed_extractions.append(f"{pdb_id}_{chain_id}")
            continue
        
        # 输出文件路径
        output_file = os.path.join(output_dir, f"{pdb_id}_{chain_id}.cif")
        
        # 提取链
        if extract_chain_from_cif(cif_file, chain_id, output_file):
            successful_extractions.append(f"{pdb_id}_{chain_id}")
        else:
            failed_extractions.append(f"{pdb_id}_{chain_id}")
    
    print(f"成功提取: {len(successful_extractions)} 个链结构")
    print(f"提取失败: {len(failed_extractions)} 个链结构")
    
    if failed_extractions:
        print(f"提取失败的链: {failed_extractions}")
    
    return successful_extractions, failed_extractions

def main():
    csv_path = os.environ.get("ATLAS_CSV_PATH", "./data/raw_data/ATLAS.csv")
    cif_save_dir = os.environ.get("CIF_SAVE_DIR", "./data/mmcifs")
    chain_output_dir = os.environ.get("CHAIN_OUTPUT_DIR", "./data/mmcifs_single_chain")
    
    print("=== 开始处理ATLAS数据集 ===")
    
    # 步骤1: 处理CSV文件，添加PDB ID和链ID列
    print("\n步骤1: 处理CSV文件...")
    df = process_atlas_csv(csv_path)
    
    # 步骤2: 下载CIF文件
    print("\n步骤2: 下载CIF文件...")
    successful_downloads, failed_downloads = download_cif_files(df, cif_save_dir)
    
    # 步骤3: 提取特定链的结构
    print("\n步骤3: 提取特定链的结构...")
    successful_extractions, failed_extractions = process_chain_extraction(
        df, cif_save_dir, chain_output_dir
    )
    
    # 总结
    print("\n=== 处理完成 ===")
    print(f"CSV文件已更新，包含 {len(df)} 行数据")
    print(f"成功下载 {len(successful_downloads)} 个CIF文件")
    print(f"成功提取 {len(successful_extractions)} 个链结构")
    print(f"输出目录: {chain_output_dir}")

if __name__ == "__main__":
    main()