# 对于跑通代码来说，只要数据格式处理对了，其他地方都不用改，因此此文件非常重要。
# 特别是对通道的处理：保留哪些通道，通道的名字是否包含在utils里的standard_1020的列表里
# 如果要对原始通道名称规范化，需要修改preprocessing_edf方法，使用ch_names = [name.split(' ')[-1].split('-')[0] for name in raw.ch_names]类似指令修改

from pathlib import Path
from shock.utils import h5Dataset
from shock.utils import preprocessing_edf

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

savePath = Path('../../processedVal')
rawDataPath = Path('../../valData')

# 使用 glob 方法从 rawDataPath 指定的目录中查找所有扩展名为 .cnt 的文件，并将其路径生成器对象赋值给 group。glob('*.cnt') 返回一个迭代器，包含符合条件的所有文件的路径。你可以通过迭代 group 来处理每个 .cnt 文件。
group = rawDataPath.rglob('*.edf')

# preprocessing parameters
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

# channel number * rsfreq
chunks = (23, rsfreq)

dataset = h5Dataset(savePath, 'dataset_tuab')

for edfFile in group:
    print(f'processing {edfFile.name}')
    eegData, chOrder = preprocessing_edf(edfFile, l_freq, h_freq, rsfreq, drop_channels, chOrder_standard)
    chOrder = [s.upper() for s in chOrder]
    # eegData = eegData[:, :-10*rsfreq] # 每个session取10s的数据
    eegData = eegData[:, :] # 每个session取完整的数据(也可选择只取前10s，但是会浪费)

    # edfFile.stem 是 pathlib.Path 对象的一个属性，它返回路径中最后一个组件的“纯粹文件名”，即去掉扩展名后的文件名。
    grp = dataset.addGroup(grpName=edfFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
