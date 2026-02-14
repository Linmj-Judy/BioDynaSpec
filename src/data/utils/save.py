
import hashlib  
import re 

def sanitize_filename(dataset_name: str) -> str:  
    # 1. 转换为小写  
    sanitized_name = dataset_name.lower()  
    # 2. 替换非字母数字字符为下划线  
    sanitized_name = re.sub(r'[^a-z0-9]+', '_', sanitized_name)  
    # 3. 删除首尾下划线  
    sanitized_name = sanitized_name.strip('_') or 'unnamed_dataset'  
    # 4. 截断过长的名称  
    max_length = 50  
    sanitized_name = sanitized_name[:max_length]  
    return sanitized_name

def generate_dataset_checkpoint_filename(data_name, data_dir_name):  
    """  
    生成基于数据目录/名称的唯一文件名  
    """  
    # 对数据目录路径进行哈希处理，确保文件名合法且唯一  
    dir_hash = hashlib.md5(data_dir_name.encode()).hexdigest()[:10] 
    sanitized_data_name = sanitize_filename(data_name) 
    return f"dataset_ckpt_{sanitized_data_name}_{dir_hash}.dill", f"feature_pt_{sanitized_data_name}_{dir_hash}", f"feature_{sanitized_data_name}_lmdb_{dir_hash}"
