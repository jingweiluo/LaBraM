import h5py
import numpy as np
from pathlib import Path

class h5Dataset:
    def __init__(self, path:Path, name:str) -> None:
        self.__name = name

        # 'a': 这是文件访问模式。在 h5py.File 中，'a' 模式表示 "append"，即：如果文件存在，则打开它进行读写操作。如果文件不存在，则创建一个新的文件。
        self.__f = h5py.File(path / f'{name}.hdf5', 'a')
    
    def addGroup(self, grpName:str):
        return self.__f.create_group(grpName)
    
    # 如果数据 data 是一个 N 维数组，那么 chunks 也应该是一个包含 N 个元素的元组，每个元素对应于数据集每一维的块大小。
    def addDataset(self, grp:h5py.Group, dsName:str, arr:np.array, chunks:tuple):
        return grp.create_dataset(dsName, data=arr, chunks=chunks)
    
    def addAttributes(self, src:'h5py.Dataset|h5py.Group', attrName:str, attrValue):
        src.attrs[f'{attrName}'] = attrValue
    
    def save(self):
        self.__f.close()
    
    @property
    def name(self):
        return self.__name

