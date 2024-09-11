import h5py

# 打开 HDF5 文件
path = "/data1/labram_data/hdf5/bci2000_dataset64.hdf5"
with h5py.File(path, 'r') as hdf:
    # 获取第一个组的名称
    first_group_name = list(hdf.keys())[0]  # 获取第一个组的名称
    first_group = hdf[first_group_name]      # 访问第一个组

    # 遍历第一个组中的所有数据集
    for dset_name in first_group:
        dset = first_group[dset_name]
        if 'chOrder' in dset.attrs:
            chOrder = dset.attrs['chOrder']
            print(f"Found 'chOrder' in dataset '{dset_name}' in group '{first_group_name}'")
            print('chOrder:', len(chOrder), chOrder)
            break
