import os  
import requests  
from tqdm import tqdm  
import pandas as pd  
import multiprocessing  
from functools import partial  
import ssl  
import urllib3  
import time  
import random 
import subprocess  


current_file = os.path.abspath(__file__)  
current_dir = os.path.dirname(current_file)
# 禁用SSL警告  
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  

file_types_subfix_map = {
    '-c': '.cif.gz',
    '-p': '.pdb.gz',
    '-A': '-assembly1.cif.gz',
    '-x': '.xml.gz',
    '-s': '-sf.cif.gz'
}

def download_pdb(args, max_retries=3, retry_delay=1):    
    """  
    下载单个PDB文件  
    
    Parameters:  
    - args: 包含(pdb_id, split, save_dir)的元组
    - max_retries: 最大重试次数  
    - retry_delay: 重试间隔  
    """  
    pdb_id, split, save_dir, file_types = args
    temp_file = os.path.join(current_dir,"tmp",f"{pdb_id}_ids.txt")
    if not os.path.exists(temp_file):  
        os.makedirs(os.path.join(current_dir,"tmp"), exist_ok=True)
        with open(temp_file, 'w') as f:  
            f.write(pdb_id)
            f.close()   
    commands = []
    for file_type in file_types:
        command = f"./batch_download.sh -f {temp_file} -o {os.path.join(save_dir,split)} {file_type}"
        commands.append(command)
    split_dir = os.path.join(save_dir, split)  
    os.makedirs(split_dir, exist_ok=True)  
    
    save_path = os.path.join(split_dir, f"{pdb_id.lower()}.pdb")  
    if os.path.exists(save_path):  
        return pdb_id, True  
    # 设置SSL上下文，忽略证书验证  
    ssl_context = ssl.create_default_context()  
    ssl_context.check_hostname = False  
    ssl_context.verify_mode = ssl.CERT_NONE  
    headers = {  
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  
    }  

    for attempt in range(max_retries): 
        try:  
            time.sleep(random.uniform(0.1, 0.5)) 
            for command in commands:
                result = subprocess.run(command, capture_output=True, shell=True, check=True)  
                time.sleep(0.05)
                
            # 检查下载是否成功  
            if result.returncode != 0:  
                print(f"Error downloading {pdb_id}: {result.stderr}") 
                return pdb_id, False 
            
            for file_type in file_types:
                gunzip_command = f"gunzip {os.path.join(split_dir, f'{pdb_id.lower()}{file_types_subfix_map[file_type]}')}"  
                subprocess.run(gunzip_command, shell=True, check=True)  

            if os.path.exists(temp_file):  
                os.remove(temp_file)             
            return pdb_id, True
        except subprocess.CalledProcessError as e:  
            print(f"Download failed for {pdb_id}: {e}")  
            return pdb_id, False  
        except Exception as e: 
            print(f"Error downloading {pdb_id} (Attempt {attempt + 1}/{max_retries}): {e}")  
            
            # 如果是最后一次尝试，仍然失败  
            if attempt == max_retries - 1:
                return pdb_id, False  
            
            # 失败后等待一段时间再重试  
            time.sleep(retry_delay * (attempt + 1))  
        if os.path.exists(temp_file):  
            os.remove(temp_file) 
    return pdb_id, False 
 
def download_pdb_bulk(csv_file, save_dir, file_types=['-p'],max_workers=None):  
    """  
    批量下载PDB文件  
    
    Parameters:  
    - csv_file: PDB ID的CSV文件  
    - save_dir: 保存目录  
    - file_types: 需要下载的文件类型列表，默认下载pdb.gz文件
    - max_workers: 并行下载的进程数  
    """  
    df = pd.read_csv(csv_file)  
    os.makedirs(save_dir, exist_ok=True)  
    if max_workers is None:  
        max_workers = multiprocessing.cpu_count()  
    
    # 准备下载参数  
    download_params = [(row['pdb'], row['split'], save_dir, file_types) for _, row in df.iterrows()]  
    
    # 使用多进程加速下载  
    with multiprocessing.Pool(processes=max_workers) as pool:  
        # 使用tqdm显示进度  
        results = list(tqdm(  
            pool.imap(download_pdb, download_params),  
            total=len(df),
            desc="Downloading PDB files"    
        ))  
    
    # 分析下载结果  
    successful = [r[0] for r in results if r[1]]  
    failed = [r[0] for r in results if not r[1]]  
    
    print(f"Successfully downloaded: {len(successful)} PDB files")  
    print(f"Failed to download: {len(failed)} PDB files")  
      
    if failed:  
        with open(os.path.join(save_dir, 'failed_downloads.txt'), 'w') as f:  
            f.write('\n'.join(failed))  

def main():  
    # CSV文件路径  
    csv_file = os.path.join(current_dir, 'Chroma_training_set.csv')
    
    # 保存PDB文件的根目录  
    save_dir = os.path.join(current_dir, 'RCSB_chorma')     
    # 可选的文件类型  
    # '-c': .cif.gz  
    # '-p': .pdb.gz  
    # '-A': -assembly1.cif.gz  
    # '-x': .xml.gz  
    # '-s': -sf.cif.gz  
    file_types = ['-c']  # 默认下载.cif.gz文件  

    # 执行批量下载  
    download_pdb_bulk(csv_file, save_dir, file_types)   

    # 统计下载结果  
    for split in ['train', 'test', 'validation']:  
        split_dir = os.path.join(save_dir, split)  
        files = os.listdir(split_dir)  
        print(f"{split.capitalize()} set: {len(files)} PDB files")  

if __name__ == '__main__':  
    main()