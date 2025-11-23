import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

from .hoss import HOSS
from .pretrain import Pretrain


__factory = {
    "HOSS": HOSS,
    "Pretrain": Pretrain,
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _, img_size = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_size


def train_pair_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    rgb_batch = [i[0] for i in batch]
    sar_batch = [i[1] for i in batch]
    batch = rgb_batch + sar_batch
    imgs, pids, camids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, img_size = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, img_size
# train_loader：带数据增强、自定义采样策略的训练集加载器（用于模型训练）；
# train_loader_normal：无数据增强（仅标准化）的训练集加载器（用于评估训练集性能）；
# val_loader：验证集加载器（包含 query 查询集 + gallery gallery 集，用于模型测试）；
# len(dataset.query)：查询集样本数量（ReID 场景核心指标计算依赖）；
# num_classes：训练集类别数（对应行人 ID 数，用于分类头构建）；
# cam_num：训练集摄像头数量（ReID 场景跨摄像头匹配需用到）。

def make_dataloader(cfg):
    # 使用 torchvision.transforms.Compose 串联多个预处理操作，训练集多数据增强
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),#调整图像尺寸为训练输入大小，interpolation=3 对应 PIL.Image.BICUBIC 插值（图像缩放更清晰）
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),#随机水平翻转，p 为翻转概率（如 0.5，提升模型泛化性）
            T.Pad(cfg.INPUT.PADDING),#图像填充，为后续随机裁剪预留边缘（避免裁剪后丢失信息）
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),#随机裁剪，裁剪到目标尺寸（增强数据多样性）
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),#标准化，按数据集统计的均值 / 标准差归一化（如 ImageNet 的 mean=[0.485,0.456,0.406]），加速模型收敛
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),#随机擦除，随机遮挡图像部分区域（增强模型对局部特征的鲁棒性，ReID 场景常用）
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )

# 验证集仅标准化
    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    print("dataset name:", cfg.DATASETS.NAMES)
    print("dataset root:", cfg.DATASETS.ROOT_DIR)


    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)#数据集工厂字典（键为数据集名称，值为数据集类），支持通过配置动态选择数据集


    train_set = ImageDataset(dataset.train, train_transforms)#自定义数据集类（需实现 __getitem__ 方法），接收数据列表（如 dataset.train 是训练集图像路径 + 标签列表）和预处理管道，返回处理后的图像和标签。
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams

    if "triplet" in cfg.DATALOADER.SAMPLER:#采样策略（sampler）：ReID 场景核心优化
        #根据 cfg.DATALOADER.SAMPLER 选择采样方式，适配不同的损失函数（Triplet Loss / Softmax Loss）
        #Triplet 采样（"triplet" in cfg.DATALOADER.SAMPLER）
        # 用途：适配 Triplet Loss（三元组损失），核心是批量中包含「相同 ID 的多个样本 + 不同 ID 的样本」，便于计算三元组距离（anchor-positive-negative）
        if cfg.MODEL.DIST_TRAIN:#分布式训练
            print("DIST_TRAIN START")
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:#单卡训练
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
            )
    elif cfg.DATALOADER.SAMPLER == "softmax":
        #Softmax 采样（cfg.DATALOADER.SAMPLER == "softmax"）
        # 用途：适配 Softmax Loss（分类损失），仅需随机打乱数据即可；
        print("using softmax sampler")
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn
        )#Softmax 采样（cfg.DATALOADER.SAMPLER == "softmax"）
         # 用途：适配 Softmax Loss（分类损失），仅需随机打乱数据即可；
    else:
        print("unsupported sampler! expected softmax or triplet but got {}".format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers, 
        collate_fn=val_collate_fn
    )#train_loader_normal 是 “无数据增强的训练集数据加载器”，核心作用是 提供 “干净、无噪声” 的训练集样本，用于训练过程中的性能监控、模型评估或特殊训练需求（如对比学习的负样本挖掘），与带数据增强的 train_loader 形成互补。
    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num


def make_dataloader_pair(cfg):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set_pair = ImageDataset(dataset.train_pair, train_transforms, pair=True)
    num_classes = dataset.num_train_pair_pids
    cam_num = dataset.num_train_pair_cams

    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    train_loader_pair = DataLoader(
        train_set_pair, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), shuffle=True, num_workers=num_workers, 
        collate_fn=train_pair_collate_fn
    )
    return train_loader_pair, num_classes, cam_num
