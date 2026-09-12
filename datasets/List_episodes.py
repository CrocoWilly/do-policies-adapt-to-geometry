import h5py

file_path = "dataset.hdf5"

with h5py.File(file_path, "r") as f:
    eps = f["data"].keys()
    print("Episodes:", len(eps))
    for k in eps:
        n = f[f"data/{k}/actions"].shape[0]
        print(k, "len =", n)
