from tqdm import tqdm

class TqdmFileReader:
    def __init__(self, file, total_size):
        self.file = file
        self.pbar = tqdm(total=total_size/1024/1024, 
                        desc="加载数据集", 
                        unit="MB")
        self.bytes_read = 0
    
    def read(self, size=-1):
        data = self.file.read(size)
        read_size = len(data)
        self.bytes_read += read_size
        self.pbar.update(read_size/1024/1024)
        return data
    
    def readline(self):
        data = self.file.readline()
        self.bytes_read += len(data)
        self.pbar.update(len(data)/1024/1024)
        return data
    
    def close(self):
        self.pbar.close()
        return self.file.close()