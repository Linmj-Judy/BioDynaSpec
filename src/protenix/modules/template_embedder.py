# Copyright 2024 自定义版权声明
"""用于小分子配体模板嵌入的模块。"""

from typing import Dict, List, NamedTuple, Optional

from alphafold3.common import base_config
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.network import modules
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class LigandFeatures(NamedTuple):
  """小分子配体模板的特征容器。"""
  # [num_templates, num_atoms]
  atom_types: jnp.ndarray
  # [num_templates, num_atoms, 3]
  atom_positions: jnp.ndarray
  # [num_templates, num_atoms]
  atom_mask: jnp.ndarray
  # [num_templates, num_atoms, num_atoms]
  bond_matrix: jnp.ndarray  # 化学键连接信息


class LigandDistogramFeaturesConfig(base_config.BaseConfig):
  """小分子配体距离图特征配置。"""
  min_bin: float = 1.0  # 小分子原子距离通常更短
  max_bin: float = 20.0  # 小分子尺寸较小
  num_bins: int = 32


def ligand_dgram_from_positions(positions, mask, config: LigandDistogramFeaturesConfig):
  """从原子位置计算小分子距离图。
  
  Args:
    positions: [num_atoms, 3] 原子坐标
    mask: [num_atoms] 原子掩码
    config: 距离图bin配置
    
  Returns:
    距离图特征 [num_atoms, num_atoms, num_bins]
  """
  # 创建距离区间
  lower_breaks = jnp.linspace(config.min_bin, config.max_bin, config.num_bins)
  upper_breaks = jnp.concatenate(
      [lower_breaks[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1
  )
  
  # 计算原子间距离平方
  mask_2d = mask[:, None] * mask[None, :]
  dist2 = jnp.sum(
      jnp.square(
          jnp.expand_dims(positions, axis=-2)
          - jnp.expand_dims(positions, axis=-3)
      ),
      axis=-1,
      keepdims=True,
  )
  
  # 生成距离图
  dgram = (dist2 > jnp.square(lower_breaks)).astype(jnp.float32) * (
      dist2 < jnp.square(upper_breaks)
  ).astype(jnp.float32)
  
  # 应用掩码
  dgram = dgram * mask_2d[..., None]
  
  return dgram


class LigandTemplateEmbedding(hk.Module):
  """小分子配体模板嵌入。"""
  
  class Config(base_config.BaseConfig):
    num_channels: int = 64
    atom_embedding_dim: int = 32  # 原子类型嵌入维度
    bond_embedding_dim: int = 16  # 键类型嵌入维度
    template_stack: modules.PairFormerIteration.Config = base_config.autocreate(
        num_layer=2,
        pair_transition=base_config.autocreate(num_intermediate_factor=2),
    )
    dgram_features: LigandDistogramFeaturesConfig = base_config.autocreate()
    max_atom_types: int = 16  # 支持的最大原子类型数
    
  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='ligand_template_embedding',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
  
  def __call__(
      self,
      query_embedding: jnp.ndarray,
      ligand_templates: LigandFeatures,
      padding_mask_2d: jnp.ndarray,
      key: jnp.ndarray,
  ) -> jnp.ndarray:
    """为一组小分子模板生成嵌入。
    
    Args:
      query_embedding: [num_atoms, num_atoms, channels] 查询嵌入
      ligand_templates: 小分子模板特征
      padding_mask_2d: [num_atoms, num_atoms] 注意力掩码
      key: 随机数生成器密钥
      
    Returns:
      尺寸为 [num_atoms, num_atoms, channels] 的嵌入
    """
    c = self.config
    num_atoms = query_embedding.shape[0]
    num_templates = ligand_templates.atom_types.shape[0]
    query_num_channels = query_embedding.shape[2]
    
    # 单独嵌入每个模板
    template_embedder = SingleLigandTemplateEmbedding(self.config, self.global_config)
    
    subkeys = jnp.array(jax.random.split(key, num_templates))
    
    def scan_fn(carry, x):
      templates, key = x
      embedding = template_embedder(
          query_embedding,
          templates,
          padding_mask_2d,
          key,
      )
      return carry + embedding, None
    
    scan_init = jnp.zeros(
        (num_atoms, num_atoms, c.num_channels), dtype=query_embedding.dtype
    )
    summed_template_embeddings, _ = hk.scan(
        scan_fn, scan_init, (ligand_templates, subkeys)
    )
    
    embedding = summed_template_embeddings / (1e-7 + num_templates)
    embedding = jax.nn.relu(embedding)
    embedding = hm.Linear(
        query_num_channels, initializer='relu', name='output_linear'
    )(embedding)
    
    return embedding


class SingleLigandTemplateEmbedding(hk.Module):
  """嵌入单个小分子模板。"""
  
  def __init__(
      self,
      config: LigandTemplateEmbedding.Config,
      global_config: model_config.GlobalConfig,
      name='single_ligand_template_embedding',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
  
  def __call__(
      self,
      query_embedding: jnp.ndarray,
      ligand_template: LigandFeatures,
      padding_mask_2d: jnp.ndarray,
      key: jnp.ndarray,
  ) -> jnp.ndarray:
    """构建单个小分子模板嵌入图。
    
    Args:
      query_embedding: [num_atoms, num_atoms, channels] 查询嵌入
      ligand_template: 单个小分子模板特征
      padding_mask_2d: [num_atoms, num_atoms] 填充掩码
      key: 随机数生成器密钥
      
    Returns:
      模板嵌入 [num_atoms, num_atoms, channels]
    """
    c = self.config
    gc = self.global_config
    dtype = query_embedding.dtype
    num_channels = c.num_channels
    
    # 构建输入特征
    def construct_input(query_embedding, template: LigandFeatures):
      to_concat = []
      
      # 1. 计算距离图特征
      atom_positions = template.atom_positions
      atom_mask = template.atom_mask
      
      # 保证坐标和掩码形状正确
      atom_positions = atom_positions * atom_mask[..., None]
      
      # 计算原子间距离图
      dgram = ligand_dgram_from_positions(
          atom_positions, atom_mask, c.dgram_features
      )
      
      # 原子掩码的2D版本
      mask_2d = atom_mask[:, None] * atom_mask[None, :]
      
      to_concat.append((dgram, 1))
      to_concat.append((mask_2d[..., None], 0))
      
      # 2. 原子类型特征
      atom_types = jax.nn.one_hot(
          template.atom_types, c.max_atom_types, axis=-1, dtype=dtype
      )
      to_concat.append((atom_types[None, :, :], 1))
      to_concat.append((atom_types[:, None, :], 1))
      
      # 3. 化学键特征
      bond_features = template.bond_matrix[..., None].astype(dtype)
      to_concat.append((bond_features, 0))
      
      # 4. 包含查询嵌入
      query_norm = hm.LayerNorm(name='query_embedding_norm')(query_embedding)
      to_concat.append((query_norm, 1))
      
      # 结合所有特征
      act = 0
      for i, (x, n_input_dims) in enumerate(to_concat):
        act += hm.Linear(
            num_channels,
            num_input_dims=n_input_dims,
            initializer='relu',
            name=f'ligand_pair_embedding_{i}',
        )(x)
      
      return act
    
    # 构建嵌入
    act = construct_input(query_embedding, ligand_template)
    
    # 应用转换层
    if c.template_stack.num_layer:
      def template_iteration_fn(x):
        return modules.PairFormerIteration(
            c.template_stack, gc, name='ligand_template_embedding_iteration'
        )(act=x, pair_mask=padding_mask_2d)
      
      template_stack = hk.experimental.layer_stack(c.template_stack.num_layer)(
          template_iteration_fn
      )
      act = template_stack(act)
    
    act = hm.LayerNorm(name='output_layer_norm')(act)
    return act


# 辅助函数：创建小分子配体模板特征
def create_ligand_template_features(
    atom_types: List[int],
    atom_positions: np.ndarray,
    bond_matrix: np.ndarray,
    num_templates: int = 1,
) -> LigandFeatures:
  """从原子类型、位置和键矩阵创建小分子模板特征。
  
  Args:
    atom_types: 原子类型ID列表
    atom_positions: 形状为 [num_atoms, 3] 的原子坐标
    bond_matrix: 形状为 [num_atoms, num_atoms] 的化学键矩阵
    num_templates: 模板数量
    
  Returns:
    LigandFeatures 对象
  """
  num_atoms = len(atom_types)
  
  # 转换为jax数组
  atom_types_array = jnp.tile(
      jnp.array(atom_types)[None, :], [num_templates, 1]
  )
  
  atom_positions_array = jnp.tile(
      jnp.array(atom_positions)[None, :, :], [num_templates, 1, 1]
  )
  
  atom_mask = jnp.ones((num_templates, num_atoms), dtype=jnp.float32)
  
  bond_matrix_array = jnp.tile(
      jnp.array(bond_matrix)[None, :, :], [num_templates, 1, 1]
  )
  
  return LigandFeatures(
      atom_types=atom_types_array,
      atom_positions=atom_positions_array,
      atom_mask=atom_mask,
      bond_matrix=bond_matrix_array,
  )