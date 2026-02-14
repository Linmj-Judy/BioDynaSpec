#!/usr/bin/env python3
"""
诊断脚本：找出为什么某些序列在 pdb_seq.csv 中存在但在 pdb_labels_seq.csv 中不存在
"""
import pandas as pd
import os
import argparse

def diagnose_missing_sequences(pdb_seq_path, pdb_labels_seq_path):
    """
    诊断序列丢失的原因
    
    Args:
        pdb_seq_path: pdb_seq.csv 文件路径
        pdb_labels_seq_path: pdb_labels_seq.csv 文件路径
    """
    print("=" * 80)
    print("序列丢失诊断分析")
    print("=" * 80)
    
    # 读取原始文件
    print(f"\n1. 读取文件...")
    print(f"   原始文件: {pdb_seq_path}")
    print(f"   标签文件: {pdb_labels_seq_path}")
    
    if not os.path.exists(pdb_seq_path):
        print(f"   ❌ 错误: {pdb_seq_path} 不存在!")
        return
    
    df_original = pd.read_csv(pdb_seq_path)
    print(f"   ✓ 原始文件包含 {len(df_original)} 行")
    
    # 检查列
    print(f"\n2. 检查列结构...")
    print(f"   原始文件列: {list(df_original.columns)}")
    
    if "mol_type" not in df_original.columns:
        print("   ❌ 错误: 原始文件缺少 'mol_type' 列!")
        return
    
    if "seq" not in df_original.columns:
        print("   ❌ 错误: 原始文件缺少 'seq' 列!")
        return
    
    # 统计 mol_type 分布
    print(f"\n3. 按 mol_type 统计...")
    mol_type_counts = df_original["mol_type"].value_counts()
    print(f"   各类型序列数量:")
    for mol_type, count in mol_type_counts.items():
        print(f"     {mol_type}: {count}")
    
    # 过滤蛋白质序列（模拟代码逻辑）
    print(f"\n4. 应用过滤条件 (mol_type == 'protein')...")
    df_protein = df_original[df_original["mol_type"] == "protein"].copy()
    print(f"   过滤后剩余: {len(df_protein)} 行")
    print(f"   丢失: {len(df_original) - len(df_protein)} 行 (非蛋白质序列)")
    
    # 检查空序列
    print(f"\n5. 检查空序列或无效序列...")
    if "seq" in df_protein.columns:
        empty_seqs = df_protein["seq"].isna() | (df_protein["seq"].str.strip() == "")
        empty_count = empty_seqs.sum()
        if empty_count > 0:
            print(f"   ⚠️  发现 {empty_count} 个空序列或无效序列")
            print(f"   这些序列的 pdb_entity_id:")
            df_protein["pdb_entity_id"] = df_protein["pdb_id"] + "_" + df_protein["entity_id"].astype(str)
            empty_pdb_ids = df_protein[empty_seqs]["pdb_entity_id"].tolist()
            for i, pdb_id in enumerate(empty_pdb_ids[:10]):  # 只显示前10个
                print(f"     {pdb_id}")
            if len(empty_pdb_ids) > 10:
                print(f"     ... 还有 {len(empty_pdb_ids) - 10} 个")
        else:
            print(f"   ✓ 没有发现空序列")
    
    # 按序列分组（模拟代码逻辑）
    print(f"\n6. 按序列分组 (groupby('seq'))...")
    df_protein["pdb_entity_id"] = df_protein["pdb_id"] + "_" + df_protein["entity_id"].astype(str)
    
    df_grouped = (
        df_protein.groupby("seq")["pdb_entity_id"]
        .apply(lambda x: ",".join(x))
        .reset_index()
    )
    print(f"   分组前: {len(df_protein)} 行")
    print(f"   分组后: {len(df_grouped)} 行")
    print(f"   合并: {len(df_protein) - len(df_grouped)} 行被合并 (相同序列的不同 pdb_entity_id)")
    
    # 如果标签文件存在，进行对比
    if os.path.exists(pdb_labels_seq_path):
        print(f"\n7. 对比标签文件...")
        df_labels = pd.read_csv(pdb_labels_seq_path)
        print(f"   标签文件包含 {len(df_labels)} 行")
        
        if "seq" in df_labels.columns:
            original_seqs = set(df_original["seq"].dropna().unique())
            labels_seqs = set(df_labels["seq"].dropna().unique())
            
            missing_in_labels = original_seqs - labels_seqs
            extra_in_labels = labels_seqs - original_seqs
            
            print(f"\n   序列对比:")
            print(f"   原始文件唯一序列数: {len(original_seqs)}")
            print(f"   标签文件唯一序列数: {len(labels_seqs)}")
            print(f"   在原始文件中但不在标签文件中: {len(missing_in_labels)}")
            print(f"   在标签文件中但不在原始文件中: {len(extra_in_labels)}")
            
            if len(missing_in_labels) > 0:
                print(f"\n   丢失的序列示例 (前10个):")
                missing_list = list(missing_in_labels)[:10]
                for seq in missing_list:
                    # 找出这些序列的 mol_type
                    seq_rows = df_original[df_original["seq"] == seq]
                    mol_types = seq_rows["mol_type"].unique()
                    pdb_ids = seq_rows["pdb_id"].unique()[:3]
                    print(f"     序列 (前50字符): {seq[:50]}...")
                    print(f"       mol_type: {mol_types}")
                    print(f"       pdb_id 示例: {pdb_ids}")
                    print()
    else:
        print(f"\n7. 标签文件不存在，跳过对比")
    
    # 总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    print("\n可能导致序列丢失的原因:")
    print("1. ✅ mol_type != 'protein' - 非蛋白质序列被过滤 (第205行)")
    print("2. ✅ 空序列或无效序列 - 可能在某些处理步骤中被移除")
    print("3. ✅ 序列分组 - 相同序列的不同 pdb_entity_id 被合并 (第209-213行)")
    print("\n建议:")
    print("- 检查原始文件中 mol_type 的分布")
    print("- 如果确实需要处理非蛋白质序列，需要修改第205行的过滤条件")
    print("- 检查是否有空序列或包含特殊字符的序列")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="诊断序列丢失原因")
    parser.add_argument("--pdb_seq_path", type=str, required=True,
                       help="pdb_seq.csv 文件路径")
    parser.add_argument("--pdb_labels_seq_path", type=str, default=None,
                       help="pdb_labels_seq.csv 文件路径 (可选)")
    
    args = parser.parse_args()
    
    diagnose_missing_sequences(args.pdb_seq_path, args.pdb_labels_seq_path)
