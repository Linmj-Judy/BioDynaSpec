import os
import json

class ConfigUtils:
    @staticmethod
    def str_to_bool(value, default=False):
        """
        将各种类型的字符串转换为布尔值
        
        支持的True值: "true", "yes", "y", "1", "on" (不区分大小写)
        支持的False值: "false", "no", "n", "0", "off" (不区分大小写)
        
        Args:
            value: 要转换的值
            default: 如果无法确定布尔值时的默认值
            
        Returns:
            转换后的布尔值
        """
        if isinstance(value, bool):
            return value
            
        if not value:  # 处理None、空字符串等
            return default
            
        if isinstance(value, str):
            value = value.lower().strip()
            
            if value in ('true', 'yes', 'y', '1', 'on', 't'):
                return True
                
            if value in ('false', 'no', 'n', '0', 'off', 'f'):
                return False
                
        try:
            return bool(int(value))
        except (ValueError, TypeError):
            pass
            
        return default
    @staticmethod
    def str_to_list(value, default=[]):
        """
        将字符串转换为列表
        
        Args:
            value: 要转换的值
            default: 如果无法确定列表时的默认值
            
        Returns:
            转换后的列表
        """
        return [item.strip() for item in value.split(',')] if value else default
    
    @staticmethod
    def str_to_dict(value, default={}):
        """
        将字符串转换为字典
        
        Args:
            value: 要转换的值
            default: 如果无法确定字典时的默认值
            
        Returns:
            转换后的字典
        """
        return json.loads(value) if value else default
    
    @staticmethod
    def get_env_bool(env_name, default=False):
        """
        从环境变量获取布尔值
        
        Args:
            env_name: 环境变量名
            default: 环境变量不存在或无法转换时的默认值
            
        Returns:
            转换后的布尔值
        """
        env_value = os.getenv(env_name)
        return ConfigUtils.str_to_bool(env_value, default)
    @staticmethod
    def get_env_list(env_name, default=[]):
        """
        从环境变量获取列表
        
        Args:
            env_name: 环境变量名    
        Returns:
            转换后的列表
        """
        env_value = os.getenv(env_name)
        return ConfigUtils.str_to_list(env_value, default)
    @staticmethod
    def get_env_dict(env_name, default={}):
        """
        从环境变量获取字典
        
        Args:
            env_name: 环境变量名
        Returns:
            转换后的字典
        """
        env_value = os.getenv(env_name)
        return ConfigUtils.str_to_dict(env_value, default)
    
    @staticmethod
    def get_env_str(env_name, default=""):
        """
        从环境变量获取字符串
        
        Args:   
            env_name: 环境变量名
        Returns:
            转换后的字符串
        """
        env_value = os.getenv(env_name)
        return env_value if env_value else default
    @staticmethod
    def get_env_int(env_name, default=0):
        """
        从环境变量获取整数
        
        Args:
            env_name: 环境变量名
        Returns:
            转换后的整数
        """
        env_value = os.getenv(env_name)
        return int(env_value) if env_value else default
    
    
class EnvConfig:
    def __init__(self,config_dict):
        self.config_dict = config_dict
        self.load_from_env()
    
    def load_from_env(self):
        """从环境变量加载配置"""
        for key, value in self.config_dict.items():
            if isinstance(value, tuple):
                if isinstance(value[1], bool):
                    self.config_dict[key] = ConfigUtils.get_env_bool(value[0], value[1])
                elif isinstance(value[1], list):
                    self.config_dict[key] = ConfigUtils.get_env_list(value[0], value[1])
                elif isinstance(value[1], dict):
                    self.config_dict[key] = ConfigUtils.get_env_dict(value[0], value[1])
                elif isinstance(value[1], str):
                    self.config_dict[key] = ConfigUtils.get_env_str(value[0], value[1])
                elif isinstance(value[1], int):
                    self.config_dict[key] = ConfigUtils.get_env_int(value[0], value[1])
    def __str__(self):
        """打印配置信息"""
        return f"Config({self.config_dict})"

