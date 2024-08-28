import h5py
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset


list_path = List[Path]

class SingleShockDataset(Dataset):
    # 功能：输入一个fileName，读取内容，按照给定的window size, stride size, start_percentage, end_percentage，将数据切割成很多个Trial
    # 实现：不对原文件进行切割，而只是获取对应的index，get的时候直接按照index去取数据（类似图书馆里，按照给定的索引取书）
    """Read single hdf5 file regardless of label, subject, and paradigm."""
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None # file下所有subject，总共的Trial数量
        self.__feature_size = None # C * W 表示每个Trial的shape

        self.__subjects = [] # 键值列表 arr = ['subject1', 'sb2', 'sb3'...]
        self.__global_idxes = [] # arr = [0, 3, 6, 9 .... 88] 一个累加序列，length和subjects数量相同，arr[n] - arr[n-1]记录的是第n个subjects下trial数量
        self.__local_idxes = [] # arr = [21, 33, 1, ...] length也与subjects数量相同，记录每个subject的第一个Trial在该subject内的startIndex
        
        self.__init_dataset()

    # 文件结构：folder -> subject1 -> eeg = C * T
    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r') # 只读形式读取HDF5文件中的数据
        self.__subjects = [i for i in self.__file] # 键值列表

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
            subject_len = self.__file[subject]['eeg'].shape[1] # 每个session的eeg信号总长度T
            # total number of samples
            total_sample_num = (subject_len-self.__window_size) // self.__stride_size + 1 # 经过滑窗后的Trial个数
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size # 第一个Trial的起始点位置
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size # 最后一个Trial的起始点位置

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length # total num of Trials

    def __getitem__(self, idx: int):
        # 入参的idx应该是指定的整个file中，一个总的trial序号
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1 # 第一步，先查看这个trial所属的subject索引
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx] # 此处返回具体的sample point的索引
        return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size] # 返回指定index的trial的data
    
    def free(self) -> None: 
        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""
    def __init__(self, file_paths: list_path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = [] # length等于subset数量，包含了所有subset的数据
        self.__length = None # 整个dataset里的trial的总数量
        self.__feature_size = None # C * W 每个Trial的shape

        self.__dataset_idxes = [] # arr = [0, 164, 352] 累加数组，表示每个subset的第一个trial的global index，也可以表示累加的trial数量。
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage, self.__end_percentage) for file_path in self.__file_paths]
        
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx]) # 表示在某个subset里的index
        return self.__datasets[dataset_idx][item_idx]
    
    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()
    
    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()
