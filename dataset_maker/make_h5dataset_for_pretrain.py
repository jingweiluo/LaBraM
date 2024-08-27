from pathlib import Path
from shock.utils import h5Dataset
from shock.utils import preprocessing_cnt

savePath = Path('path/to/your/save/path')
rawDataPath = Path('path/to/your/raw/data/path')

# 使用 glob 方法从 rawDataPath 指定的目录中查找所有扩展名为 .cnt 的文件，并将其路径生成器对象赋值给 group。glob('*.cnt') 返回一个迭代器，包含符合条件的所有文件的路径。你可以通过迭代 group 来处理每个 .cnt 文件。
group = rawDataPath.glob('*.cnt')

# preprocessing parameters
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

# channel number * rsfreq
chunks = (62, rsfreq)

dataset = h5Dataset(savePath, 'dataset')
for cntFile in group:
    print(f'processing {cntFile.name}')
    eegData, chOrder = preprocessing_cnt(cntFile, l_freq, h_freq, rsfreq)
    chOrder = [s.upper() for s in chOrder]
    eegData = eegData[:, :-10*rsfreq] # 每个session取10s的数据

    # cntFile.stem 是 pathlib.Path 对象的一个属性，它返回路径中最后一个组件的“纯粹文件名”，即去掉扩展名后的文件名。
    grp = dataset.addGroup(grpName=cntFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
