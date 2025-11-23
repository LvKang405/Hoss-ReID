# encoding: utf-8
import glob
import os.path as osp
from .bases import BaseImageDataset
import os

class HOSS(BaseImageDataset):
    """
    HOSS dataset
    """
    dataset_dir = 'HOSS'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(HOSS, self).__init__()
        print('HOSS->init()root='+root)
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 拼接其他路径
        root = os.path.join(current_dir, root)
        print('相对路径变绝对路径HOSS->init()root='+root)
        # 4. 解析为最终的绝对路径
        root = os.path.abspath(root)
        print('解析绝对路径HOSS->init()root='+root)
        print('HOSS->init()self.dataset_dir='+self.dataset_dir)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train, train_pair = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> HOSS ReID Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)
            if train_pair is not None:
                print("Number of RGB-SAR pair: {}".format(len(train_pair)))
                print("  ----------------------------------------")

        self.train = train
        self.train_pair = train_pair
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(self.train_pair)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def get_imagedata_info_pair(self, data):
        pids, cams, tracks = [], [], []

        for img in data:
            for _, pid, camid, trackid in img:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 输出结果
        print("当前工作目录：", current_dir)
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(img_path.split('\\')[-1].split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(img_path.split('\\')[-1].split('_')[0])
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split('\\')[-1].split('_')[-1] == 'RGB.tif' else 1
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir_train(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))

        RGB_paths = [i for i in img_paths if i.endswith('RGB.tif')]
        pid2sar = {}

        pid_container = set()
        for img_path in sorted(img_paths):
            print('img_path='+img_path)
            print('img_path.split("/")[-1].split("_")[0]='+str(img_path.split('/')[-1].split('_')[0]))
            print('img_path.split("/")[-1]='+img_path.split('/')[-1])
            pid = int(img_path.split('\\')[-1].split('_')[0])
            print('pid='+str(pid))
            pid_container.add(pid)
            if img_path.endswith('SAR.tif'):
                if pid not in pid2sar:
                    pid2sar[pid] = [img_path]
                else:
                    pid2sar[pid].append(img_path)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            print('img_path='+img_path)
            pid = int(img_path.split('\\')[-1].split('_')[0])
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split('\\')[-1].split('_')[-1] == 'RGB.tif' else 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        dataset_pair = []
        for img_path in sorted(RGB_paths):
            pid = int(img_path.split('\\')[-1].split('_')[0])
            if pid not in pid2sar.keys():
                continue
            for sar_path in pid2sar[pid]:
                dataset_pair.append([(img_path, self.pid_begin + pid, 0, 1),
                                     (sar_path, self.pid_begin + pid, 1, 1)])

        return dataset, dataset_pair