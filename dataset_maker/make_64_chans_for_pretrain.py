import mne
from pathlib import Path
from shock.utils import h5Dataset

file_type = ".edf"
num_chans = 64
savePath = Path('/data1/labram_data/hdf5/')
filename = "bci2000_eval_dataset64"
rawDataPath = Path('/data1/labram_data/BCI2000_eval/')
group = rawDataPath.rglob('*.edf')

# file_type = ".bdf"
# num_chans = 64
# savePath = Path('/data1/labram_data/hdf5/')
# filename = "raweegdata_dataset64"
# rawDataPath = Path('/data1/labram_data/Raw EEG Data/')
# group = rawDataPath.rglob('*.bdf')

standard_64_channels = [
    'FP1', 'FPZ', 'FP2',    # 前额区域
    'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',   # 额极前区域
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',   # 额区
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',   # 额颞区、额中央区
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',   # 中央区
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',   # 顶中央区
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',   # 顶区
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',   # 枕顶区
    'O1', 'OZ', 'O2'    # 枕区
]

# preprocessing parameters
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

# channel number * rsfreq
chunks = (num_chans, rsfreq)

def preprocessing_edf(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq: int = 200, standard_channels: list = None):
    # 读取 edf 文件
    raw = mne.io.read_raw_edf(edfFilePath, preload=True)
    # raw = mne.io.read_raw_bdf(edfFilePath, preload=True)

    # 如果采样率小于160Hz则跳过处理
    if raw.info['sfreq'] < 160:
        return None, None
    
    # raw.rename_channels({ch: ch.strip('.').upper() for ch in raw.ch_names})
    # drop_chans = [ch for ch in raw.ch_names if ch not in standard_64_channels]
    # print('丢弃的通道', drop_chans)
    # raw.drop_channels(drop_chans)

    # if standard_channels is not None and len(standard_channels) == len(raw.ch_names):
    #     try:
    #         raw.reorder_channels(standard_channels)
    #     except:
    #         return None, ['a']

    # 过滤
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw = raw.notch_filter(50.0)
    
    # 重采样
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    # 处理通道名称：去掉头尾多余的.然后大写
    chOrder = [s.strip(".").upper() for s in raw.ch_names]

    return eegData, chOrder


# 处理每个 EEG 文件
dataset = h5Dataset(savePath, filename)
for eegFile in group:
    print(f'processing {eegFile.name}')
    
    # 调用 preprocessing_edf 并获取返回值
    eegData, chOrder = preprocessing_edf(eegFile, l_freq, h_freq, rsfreq, standard_64_channels)

    # 如果返回值为 None，则跳过该文件
    if eegData is None:
        print(f'Skipping {eegFile.name} due to insufficient sampling rate.')
        continue
    else:
        print("通道数", len(chOrder), chOrder)


    # 处理剩余的逻辑
    # 如果需要限制数据长度，可以使用以下代码进行截断
    # eegData = eegData[:, :-10*rsfreq] # 每个session取10s的数据

    # eegFile.stem 是 pathlib.Path 对象的一个属性，它返回路径中最后一个组件的“纯粹文件名”，即去掉扩展名后的文件名。
    grp = dataset.addGroup(grpName=eegFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # 添加数据集属性
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

# 保存数据集
dataset.save()


# def read_edf(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200, drop_channels: list=None, standard_channels: list=None):
#     # reading edf
#     raw = mne.io.read_raw_edf(edfFilePath, preload=True)
#     # raw = mne.io.read_raw_bdf(edfFilePath, preload=True)
#     eegData = raw.get_data(units='uV')
#     sampling_freq = raw.info['sfreq']
#     times = raw.times[-1]
#     return eegData, raw.ch_names, sampling_freq, times

# duration = 0
# # 读取通道信息
# for index, eegFile in enumerate(group):
#     print(f'processing {eegFile.name}')
#     eegData, chOrder, sampling_freq, times = read_edf(eegFile, l_freq, h_freq, rsfreq)
#     print(len(chOrder), chOrder)

#     if index == 5:
#         print(len(chOrder), chOrder)
#     duration += times

# print("数据集总时长", duration // 3600)