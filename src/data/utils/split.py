from typing import Callable, Optional, Tuple, List  
import random  

# from src.utils.process_single_file import process_single_file  
# from src.utils.generate_prmtop import generate_prmtop_files_batch  


def train_test_split(  
    data: List, train_size: float, test_size: float, random_state: Optional[int] = None  
) -> Tuple[List, List]:  
    """  
    手动实现 train_test_split，将数据集划分为训练集和测试集。  

    Args:  
        data: 原始数据集（一个列表）。  
        train_size: 训练集比例。  
        test_size: 测试集比例。  
        random_state: 随机数种子，用于保证可重复性。  

    Returns:  
        一个包含训练集和测试集的元组。  
    """  
    if not 0 <= train_size <= 1 and not 0 <= test_size <= 1 and not 0 <= train_size + test_size <= 1:  
        raise ValueError("train_size and test_size must be between 0 and 1, and their sum must also be between 0 and 1")  

    if random_state is not None:  
        random.seed(random_state)  

    # 创建数据集的副本，避免修改原始数据  
    shuffled_data = data[:]  
    random.shuffle(shuffled_data)  

    # 计算划分点  
    split_index = int(len(shuffled_data) * (1 - test_size))  

    # 划分数据集  
    train_data = shuffled_data[:split_index]  
    test_data = shuffled_data[split_index:]  

    return train_data, test_data  


def train_valid_test_split(  
    data: List,  
    train_size: float,  
    val_size: float,  
    test_size: float,  
    random_state: Optional[int] = None,  
) -> Tuple[List, List, List]:  
    """  
    手动实现 train_valid_test_split，将数据集划分为训练集、验证集和测试集。  

    Args:  
        data: 原始数据集（一个列表）。  
        val_size: 验证集比例。  
        test_size: 测试集比例。  
        random_state: 随机数种子，用于保证可重复性。  

    Returns:  
        一个包含训练集、验证集和测试集的元组。  
    """  
    if not 0 <= train_size <= 1 and not 0 <= val_size <= 1 and not 0 <= test_size <= 1 and not 0 <= train_size + val_size + test_size <= 1:  
        raise ValueError("train_size, val_size, and test_size must be between 0 and 1, and their sum must also be between 0 and 1")  

    if random_state is not None:  
        random.seed(random_state)  

    shuffled_data = data[:]  
    random.shuffle(shuffled_data)  

    # 计算划分点  
    test_split_index = int(len(shuffled_data) * (1 - test_size))  
    val_split_index = int(len(shuffled_data[:test_split_index]) * (1 - val_size / (1-test_size)))  

    # 划分数据集  
    train_data = shuffled_data[:val_split_index]  
    val_data = shuffled_data[val_split_index:test_split_index]  
    test_data = shuffled_data[test_split_index:]  

    return train_data, val_data, test_data