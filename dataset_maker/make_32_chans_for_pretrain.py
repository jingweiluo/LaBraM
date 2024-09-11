import scipy.io as sio
from pathlib import Path
from shock.utils import h5Dataset
import numpy as np
import mne

# 配置
file_type = ".mat"
num_chans = 32
savePath = Path('/data1/labram_data/hdf5/')
filename = "target_dataset32"
rawDataPath = Path('/data1/labram_data/bi2015a_Target_vs_nonTarget/')
group = rawDataPath.rglob('*.mat')

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF', 'NAS', 'LVEOG', 'RVEOG', 'LHEOG', 'RHEOG', 'NFpz', 'Status', 'M1', 'M2']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

# 预处理参数
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

# 通道数 * 采样率
chunks = (num_chans, rsfreq)

def preprocessing_mat(matFilePath, l_freq=0.1, h_freq=75.0, sfreq: int = 200, drop_channels: list = None, standard_channels: list = None):
    # 读取 .mat 文件
    mat = sio.loadmat(matFilePath)

    # 从 .mat 文件中提取数据和通道名（假设通道名存储在 'channels' 字段）
    try:
        eegData = mat['eegData']  # 假设 EEG 数据存储在 'eegData' 字段
        ch_names = [str(ch[0]) for ch in mat['channels'][0]]  # 提取通道名称，假设通道名称存储在 'channels' 字段
    except KeyError:
        print(f"File {matFilePath} does not contain expected data structure.")
        return None, None

    # 处理丢弃的通道
    if drop_channels is not None:
        indices_to_drop = [i for i, ch in enumerate(ch_names) if ch in drop_channels]
        eegData = np.delete(eegData, indices_to_drop, axis=0)
        ch_names = [ch for ch in ch_names if ch not in drop_channels]

    # 处理标准化通道顺序
    if standard_channels is not None and len(standard_channels) == len(ch_names):
        try:
            ordered_indices = [ch_names.index(ch) for ch in standard_channels if ch in ch_names]
            eegData = eegData[ordered_indices, :]
            ch_names = [ch_names[i] for i in ordered_indices]
        except ValueError:
            print("Standard channels do not match with the channels in the file.")
            return None, ['a']

    # 假设滤波和重采样是通过外部工具完成的，或者在这里添加额外的滤波和重采样处理
    # 如果有需要，可以加上滤波和重采样代码
    # eegData = mne.filter.filter_data(eegData, sfreq, l_freq, h_freq)
    # eegData = mne.filter.resample(eegData, down=sfreq / new_sfreq, npad='auto')

    # 处理通道名称：去掉头尾多余的 "." 并大写
    chOrder = [ch.strip(".").upper() for ch in ch_names]

    return eegData, chOrder


# 处理每个 .mat 文件
dataset = h5Dataset(savePath, filename)
for matFile in group:
    print(f'processing {matFile.name}')
    if matFile.name == 'Header.mat':
        continue
    
    # 调用 preprocessing_mat 并获取返回值
    eegData, chOrder = preprocessing_mat(matFile, l_freq, h_freq, rsfreq, drop_channels, chOrder_standard)

    print("通道数", len(chOrder))
    
    # 如果返回值为 None，则跳过该文件
    if eegData is None:
        print(f'Skipping {matFile.name} due to insufficient data or structure.')
        continue

    # 如果需要限制数据长度，可以使用以下代码进行截断
    # eegData = eegData[:, :-10*rsfreq] # 每个session取10s的数据

    # matFile.stem 是 pathlib.Path 对象的一个属性，它返回路径中最后一个组件的“纯粹文件名”，即去掉扩展名后的文件名。
    grp = dataset.addGroup(grpName=matFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # 添加数据集属性
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

# 保存数据集
dataset.save()
