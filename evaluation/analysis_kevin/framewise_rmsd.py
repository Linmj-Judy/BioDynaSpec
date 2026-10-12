import mdtraj as md
import numpy as np
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
import multiprocessing
import traceback
from functools import partial


def calculate_rmsd_for_protein(pdb_name, root_reference, root_net1, root_net2):
    """计算单个蛋白质的参考轨迹与两个网络轨迹之间的RMSD"""
    try:
        # 构建文件路径
        ref_pdb = os.path.join(root_reference, pdb_name, f"{pdb_name}.pdb")
        ref_xtc = os.path.join(root_reference, pdb_name, f"{pdb_name}_R3.xtc")
        net1_pdb = os.path.join(root_net1, f"{pdb_name}.pdb")
        net2_pdb = os.path.join(root_net2, f"{pdb_name}.pdb")

        # 检查文件是否存在
        if not os.path.exists(ref_pdb):
            return pdb_name, f"错误: 文件不存在 - {ref_pdb}"
        if not os.path.exists(ref_xtc):
            return pdb_name, f"错误: 文件不存在 - {ref_xtc}"
        if not os.path.exists(net1_pdb):
            return pdb_name, f"错误: 文件不存在 - {net1_pdb}"
        if not os.path.exists(net2_pdb):
            return pdb_name, f"错误: 文件不存在 - {net2_pdb}"

        # 1. 加载参考轨迹
        ref_traj = md.load_xtc(ref_xtc, top=ref_pdb)

        # 2. 创建时间轴 (单位: ps)
        ref_time = np.arange(0, len(ref_traj) * 100, 100)  # 100ps间隔
        net_time = np.arange(0, 250 * 400, 400)  # 400ps间隔

        # 3. 下采样参考轨迹以匹配网络轨迹时间点
        indices = []
        for t in net_time:
            idx = np.abs(ref_time - t).argmin()
            indices.append(idx)
        ref_subsampled = ref_traj[indices]

        # 4. 加载网络轨迹
        net1_traj = md.load_pdb(net1_pdb)
        net2_traj = md.load_pdb(net2_pdb)

        # 5. 检查轨迹长度
        if len(ref_subsampled) != 250:
            return (
                pdb_name,
                f"错误: 下采样后参考轨迹长度应为250帧，实际为{len(ref_subsampled)}帧",
            )
        if len(net1_traj) != 250:
            return pdb_name, f"错误: 网络1轨迹长度应为250帧，实际为{len(net1_traj)}帧"
        if len(net2_traj) != 250:
            return pdb_name, f"错误: 网络2轨迹长度应为250帧，实际为{len(net2_traj)}帧"

        # 6. 选择CA原子进行叠加
        ref_ca_atoms = ref_subsampled.topology.select("name CA")
        net1_ca_atoms = net1_traj.topology.select("name CA")
        net2_ca_atoms = net2_traj.topology.select("name CA")

        # 7. 叠加到参考轨迹的第一帧
        ref_frame = ref_subsampled[0]
        ref_aligned = ref_subsampled.superpose(ref_frame, atom_indices=ref_ca_atoms)
        net1_aligned = net1_traj.superpose(ref_frame, atom_indices=net1_ca_atoms)
        net2_aligned = net2_traj.superpose(ref_frame, atom_indices=net2_ca_atoms)

        # 8. 计算逐帧RMSD (修正部分)
        rmsd_net1 = []
        rmsd_net2 = []

        # 对每一帧计算RMSD
        for i in range(len(ref_aligned)):
            # 计算网络1轨迹第i帧与参考轨迹第i帧的RMSD
            r1 = md.rmsd(
                net1_aligned[i],
                ref_aligned[i],
                atom_indices=net1_ca_atoms,
                ref_atom_indices=ref_ca_atoms,
                frame=0,  # 使用当前帧自身坐标
            )[0]

            # 计算网络2轨迹第i帧与参考轨迹第i帧的RMSD
            r2 = md.rmsd(
                net2_aligned[i],
                ref_aligned[i],
                atom_indices=net2_ca_atoms,
                ref_atom_indices=ref_ca_atoms,
                frame=0,  # 使用当前帧自身坐标
            )[0]

            rmsd_net1.append(r1)
            rmsd_net2.append(r2)

        rmsd_net1 = np.array(rmsd_net1)
        rmsd_net2 = np.array(rmsd_net2)

        # 9. 计算不同时间段的平均RMSD
        def calc_rmsd_stats(rmsd_values):
            return {
                "R50": np.mean(rmsd_values[:50]),
                "R100": np.mean(rmsd_values[:100]),
                "R150": np.mean(rmsd_values[:150]),
                "R200": np.mean(rmsd_values[:200]),
                "R250": np.mean(rmsd_values),
            }

        return pdb_name, {
            "net1": calc_rmsd_stats(rmsd_net1),
            "net2": calc_rmsd_stats(rmsd_net2),
        }

    except Exception as e:
        error_msg = f"处理 {pdb_name} 时出错: {str(e)}\n{traceback.format_exc()}"
        return pdb_name, error_msg


def process_wrapper(
    pdb_name, root_reference, root_net1, root_net2, results_queue, errors_queue
):
    """包装函数用于多进程，将结果和错误放入队列"""
    try:
        result = calculate_rmsd_for_protein(
            pdb_name, root_reference, root_net1, root_net2
        )
        if isinstance(result[1], str):  # 错误信息
            errors_queue.put((pdb_name, result[1]))
        else:  # 成功结果
            results_queue.put((pdb_name, result[1]))
    except Exception as e:
        errors_queue.put(
            (
                pdb_name,
                f"处理 {pdb_name} 时发生未捕获的异常: {str(e)}\n{traceback.format_exc()}",
            )
        )


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Frame-wise C-alpha RMSD (paper metric R_s: mean RMSD over the first "
            "50/100/150/200/250 frames) between the reference MD ensemble and two "
            "generated ensembles."
        )
    )
    p.add_argument("--root-reference", default="/public/home/lingwang/lmj/kevin/MD_raw_data/ATLAS",
                   help="Reference MD root; expects <name>/<name>.pdb and <name>_R3.xtc")
    p.add_argument("--root-net1", default="/public/home/lingwang/lmj/kevin/file_trans/mdgen_results/atlas_R3",
                   help="Generated ensemble #1 root; expects <name>.pdb (250 frames)")
    p.add_argument("--root-net2", default="/public/home/lingwang/lmj/kevin/file_trans/2025-07-21-p2dflow-R3",
                   help="Generated ensemble #2 root; expects <name>.pdb (250 frames)")
    p.add_argument("--split-csv", default="/public/home/lingwang/lmj/kevin/MD_data/split/atlas_test.csv",
                   help="CSV whose first column lists the protein names to process")
    p.add_argument("--output", default="rmsd_results.txt",
                   help="Where to write the results table")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Worker processes (default: cpu_count() // 2)")
    return p


def main(args):
    # 配置路径
    root_reference = args.root_reference
    root_net1 = args.root_net1
    root_net2 = args.root_net2

    # 蛋白质列表 - 实际使用时应替换为真实列表
    protein_names = pd.read_csv(args.split_csv).iloc[:, 0]

    # 创建多进程队列
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    errors_queue = manager.Queue()

    # 创建进程池
    num_cores = args.num_workers or (multiprocessing.cpu_count() // 2)
    pool = multiprocessing.Pool(processes=num_cores)

    # 启动所有任务
    for pdb_name in protein_names:
        pool.apply_async(
            process_wrapper,
            args=(
                pdb_name,
                root_reference,
                root_net1,
                root_net2,
                results_queue,
                errors_queue,
            ),
        )

    # 关闭进程池，不再接受新任务
    pool.close()

    # 存储结果和错误
    all_results = {}
    all_errors = {}

    # 使用tqdm显示进度条
    with tqdm(total=len(protein_names), desc="分析蛋白质轨迹", file=sys.stdout) as pbar:
        # 定期检查队列并更新进度
        completed = 0
        while completed < len(protein_names):
            # 检查结果队列
            while not results_queue.empty():
                pdb_name, result = results_queue.get()
                all_results[pdb_name] = result
                completed += 1
                pbar.update(1)

            # 检查错误队列
            while not errors_queue.empty():
                pdb_name, error = errors_queue.get()
                all_errors[pdb_name] = error
                completed += 1
                pbar.update(1)
                # 立即打印错误信息到stderr，避免被进度条阻塞
                print(f"\n错误: {error}", file=sys.stderr)

            # 短暂休眠避免CPU占用过高
            # pool.join(0.1)

    # 确保所有进程完成
    pool.join()

    # 最后再检查一次队列，以防有遗漏
    while not results_queue.empty():
        pdb_name, result = results_queue.get()
        all_results[pdb_name] = result

    while not errors_queue.empty():
        pdb_name, error = errors_queue.get()
        all_errors[pdb_name] = error
        print(f"\n错误: {error}", file=sys.stderr)

    # 输出错误摘要
    if all_errors:
        print(f"\n处理完成，但有 {len(all_errors)} 个蛋白质出错:", file=sys.stderr)
        for pdb_name, error in all_errors.items():
            print(f"{pdb_name}: {error.splitlines()[0]}", file=sys.stderr)

    # 计算总体平均值
    avg_net1 = {"R50": 0, "R100": 0, "R150": 0, "R200": 0, "R250": 0}
    avg_net2 = {"R50": 0, "R100": 0, "R150": 0, "R200": 0, "R250": 0}
    count = 0

    # 写入结果文件
    with open(args.output, "w") as f:
        # 表头
        f.write(
            "Protein\tNet1_R50\tNet1_R100\tNet1_R150\tNet1_R200\tNet1_R250\t"
            "Net2_R50\tNet2_R100\tNet2_R150\tNet2_R200\tNet2_R250\n"
        )

        # 每个蛋白质的结果
        for pdb_name in protein_names:
            if pdb_name in all_results:
                results = all_results[pdb_name]
                net1 = results["net1"]
                net2 = results["net2"]

                # 写入当前蛋白质结果
                f.write(
                    f"{pdb_name}\t{net1['R50']:.4f}\t{net1['R100']:.4f}\t{net1['R150']:.4f}\t{net1['R200']:.4f}\t{net1['R250']:.4f}\t"
                    f"{net2['R50']:.4f}\t{net2['R100']:.4f}\t{net2['R150']:.4f}\t{net2['R200']:.4f}\t{net2['R250']:.4f}\n"
                )

                # 累加计算平均值
                for key in avg_net1:
                    avg_net1[key] += net1[key]
                    avg_net2[key] += net2[key]
                count += 1
            else:
                # 对于出错或缺失的蛋白质，写入占位符
                f.write(
                    f"{pdb_name}\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\n"
                )

        # 计算总体平均值
        if count > 0:
            for key in avg_net1:
                avg_net1[key] /= count
                avg_net2[key] /= count

            # 写入平均值
            f.write("\n")  # 空行分隔
            f.write(
                f"Average ({count} proteins)\t{avg_net1['R50']:.4f}\t{avg_net1['R100']:.4f}\t{avg_net1['R150']:.4f}\t{avg_net1['R200']:.4f}\t{avg_net1['R250']:.4f}\t"
                f"{avg_net2['R50']:.4f}\t{avg_net2['R100']:.4f}\t{avg_net2['R150']:.4f}\t{avg_net2['R200']:.4f}\t{avg_net2['R250']:.4f}\n"
            )

    print(f"\n分析完成! 成功处理 {count} 个蛋白质，{len(all_errors)} 个出错")
    print(f"结果已保存到 {args.output}")


if __name__ == "__main__":
    # 确保Windows系统上多进程正常工作
    multiprocessing.freeze_support()
    main(build_parser().parse_args())
