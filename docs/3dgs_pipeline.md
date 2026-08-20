# TinyNav 地图 → 3DGS

实测于 `office_big_table_slow`（2321 关键帧 / 43 m，降采样到 1200 帧），相机 Insight9。
产物 `splat_vs45.ply`，96,315 个高斯。

## 一、怎么跑

### 1. 建图

```bash
uv run python tinynav/core/build_map_node.py \
    --bag-file rosbags/office_big_table_slow \
    --map-save-path output/map_big_table_slow
# 另一个终端：uv run python tinynav/core/perception_node.py
```

建完确认：关键帧时间戳落在 bag 的消息头时间戳范围内、RGB 亮度不接近 0。
外参没写进去（`T_rgb_to_infra1.npy` 是 `None`）就用
`tool/extract_rgb_extrinsic.py --bag-dir <bag> --map-dir <map>` 从 `/tf_static` 补。

### 2. 转换

```bash
uv run python tool/convert_to_nerf_format.py --map-dir output/map_big_table_slow \
    --min-sharpness 50 --drop-blurriest 0.10 --max-pose-deviation 0.5 \
    --max-frames 1200 --pixel-stride 4 --max-points 1500000 \
    --max-depth 5.0 --depth-maps --depth-max 6.0
```

`--max-frames` 按内存定：1200 张全分辨率图缓存约 7.5 GB（GPU 机只有 31 GB）。
`--max-depth 5.0` 是种子云截断，双目误差按 z² 增长，远处的点是噪声。
`--depth-maps` 产出的深度图**供剪枝用，不是监督用**。

### 3. 位姿（必须做，见第二节）

```bash
uv run python tool/colmap_ba_refine.py export   --map-dir output/map_big_table_slow
uv run python tool/colmap_ba_refine.py features --map-dir output/map_big_table_slow
uv run python tool/colmap_ba_refine.py sfm      --map-dir output/map_big_table_slow
uv run python tool/colmap_ba_refine.py write    --map-dir output/map_big_table_slow
uv run python tool/colmap_ba_refine.py validate --map-dir output/map_big_table_slow
```

产出 `transforms_ba.json` + `sparse_pc_ba.ply`，`poses.npy` 和 `transforms.json` 不动。
1200 张图：匹配几小时 + 重建一夜。图多于约 800 张想快些就
`features --matcher sequential --overlap 20`。

### 4. 同步到 GPU 机

```bash
REMOTE=dm@192.168.10.46
R=/mnt/data/workspace/qichen/tinynav
MAP=output/map_big_table_slow

rsync -avhP scripts/run_3dgs_generation.sh $REMOTE:$R/scripts/
rsync -avhP tool/prune_splat.py $REMOTE:$R/tool/
rsync -avhP $MAP/transforms_ba.json $REMOTE:$R/$MAP/
rsync -avhP $MAP/sparse_pc_ba.ply $REMOTE:$R/$MAP/
rsync -avhP $MAP/images/ $REMOTE:$R/$MAP/images/
rsync -avhP $MAP/depths_rgb/ $REMOTE:$R/$MAP/depths_rgb/
```

只在 `depth_supervision=true` 时才需要额外传 `tool/train_depth_splatfacto.py`。
1200 帧的 `images/` 约 1 GB、`depths_rgb/` 约 2.5 GB。

### 5. 训练

```bash
map_path=.../output/map_big_table_slow \
  data_path=.../output/map_big_table_slow/transforms_ba.json \
  cuda_devices=1 output_path=.../nerf_best \
  bash scripts/run_3dgs_generation.sh --remote
```

脚本默认值即验证过的配方，不用传参数。10 万步约 3 小时，结束自动剪枝一次。

### 6. 剪枝

训练结束脚本已自动剪过一次（宽松档）。要更干净的，从原始 ply 重新剪 —— 两套预设，按用途选：

```bash
rsync -avhP $REMOTE:$R/$MAP/nerf_best/splat.ply $MAP/nerf_best/splat.ply
```

**A. 覆盖全程**（`splat_best_clean.ply`，123,111 个高斯）—— 导航、POI、走进去看：

```bash
uv run python tool/prune_splat.py --map-dir $MAP --splat $MAP/nerf_best/splat.ply \
    --max-height 0.0 --require-support-beyond 1.0 --min-support 10 \
    --isolation-radius 0.08 --keep-largest-cluster \
    --output $MAP/splat_best_clean.ply
```

保留整条 43 m 轨迹上的几何，只按高度切掉天花板方向（`--max-height` 取相机高度中位数+0.6 m 左右，这份数据相机在 −0.99…−0.22、中位 −0.62，所以用 0.0）。

**B. 任意视角都干净**（`splat_vs45.ply`，96,315 个高斯）—— 对外展示、俯视浏览：

```bash
uv run python tool/prune_splat.py --map-dir $MAP --splat $MAP/nerf_best/splat.ply \
    --min-view-spread 45 --require-support-beyond 1.0 --min-support 10 \
    --isolation-radius 0.08 --keep-largest-cluster \
    --output $MAP/splat_vs45.ply
```

用视角张开度替代高度切。代价是删掉的 66.6% 里有大量真实表面（见第二节），所以**不要拿它做导航或几何量测**，它是展示物。

看结果：

```bash
uv run python tool/poi_editor.py --tinynav-map-path $MAP --splat-path $MAP/splat_vs45.ply
```

**两个预设都只在原始导出的 ply 上跑一次。** 剪枝破坏连通性，对已剪过的文件再跑一遍，连通性规则会删掉整片区域（实测某个模型的 51%）。

---

## 二、改了什么，为什么

### 位姿：从 VO 换成 COLMAP SfM

| 位姿 | 重投影误差 | 深度一致性 |
|---|---|---|
| VO（`poses.npy`） | 1.711 px | 40.5% |
| SfM | 1.176 px | **11.4%** |

VO 局部好、全局漂移（43 m 走出 20.8 cm），3DGS 要亚像素自洽，于是同一表面每帧落在不同位置 → 重影和发虚。**这是整轮里唯一带来质变的改动**，前面四轮调训练参数都只有微小改善。

`colmap_ba_refine.py validate` 用双目深度当独立裁判：重投影误差本身分不出"修好了位姿"和"把场景掰弯了"——两者都会让它下降。深度从未参与位姿计算，所以它的判定是独立的。

### 训练参数：为坏位姿加的东西，位姿修好后要撤掉

| 改动 | 为什么 |
|---|---|
| 全分辨率 + 10 万步 + LR/密化日程同步拉长 | **保留**。原厂 3 万步日程之后每一步对几何都是 no-op |
| `use_bilateral_grid=True` | **保留**。RGB 自动曝光，没有逐图校正时模型把不同曝光平均成灰雾 |
| `camera-optimizer SO3xR3` lr 1e-3 | **保留**。吸收残余位姿误差 |
| `sh_degree` 1 → 3 | 降到 1 只在位姿噪声大到让 3 阶 SH 把误差解释成视角相关高光时才有用 |
| `use_scale_regularization` → **关** | 它把每个高斯往 `max_gauss_ratio` 捏而不是修剪离群值（实测全模型各向异性堆在上限）。墙面高斯本该是薄扁盘，压成球会让桌面和人脸发钝。针状高斯留到导出后精确剪 |
| 深度监督 → **关** | VO 位姿时减轻重影；SfM 位姿下有害——双目深度约 3% 误差（1.4 m 处 4 cm）比多视角光度更差，深度损失把高斯从图像认定的位置拉走 |

**深度的价值在训练之后**：当监督它和光度竞争并且会输；当剪枝判据（空间雕刻、支撑度）它是决定性的。

### 剪枝：八条规则

`splat_vs45.ply` 里各规则的命中比例：

| 规则 | 判据 | 命中 |
|---|---|---|
| **narrow-views** | 观测视线的**角度张开度** < 45°。只从一个方向擦过的表面，形状在别的方向从未被约束，从那些方向看必然是薄膜——这是"碎片"的数学定义，也是主判据 | 66.6% |
| unconfirmed | 1 m 外、少于 10 个视角的深度确认过该位置 | 51.3% |
| off-cluster | 不属于最大连通体。抓"内部密集所以躲过孤立性判据"的碎云 | 19.9% |
| needle | 中轴/长轴 < 0.2 且长轴 > 5 cm。**长短轴比值分不出针和墙面扁盘，中轴可以** | 18.6% |
| isolated | 第 8 近邻 > 8 cm（真实表面中位 3.2 cm） | 11.8% |
| free-space | 在 ≥3 个视角里位于实测表面之前且从不落在表面上 —— 空间雕刻，原理同 TSDF | 6.5% |
| faint / oversized / unobserved | 雾、跨房间糊块、从未被看到 | 5.5% |
| above-ceiling / ROI 盒子 | **不是算法判据**，是声明交付范围，仅用于紧凑展示物 | 未用 |

这份数据的视角张开度中位数只有 **27.9°**——单向走一趟，一半几何本就只从窄角度被看到，所以 45° 阈值删掉的 66.6% 里有大量真实表面。公开数据集上同样的算法更干净，不是算法更好，而是他们覆盖充分、narrow-view 只占几个百分点。

