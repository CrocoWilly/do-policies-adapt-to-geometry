# List the structure of an HDF5 file.
import h5py
from rich import print

file_path = "dataset.hdf5"   # change this to your path
# file_path = "official_dataset.hdf5"

def print_tree(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[green]{name}[/green] shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"[blue]{name}[/blue]")

with h5py.File(file_path, "r") as f:
    f.visititems(print_tree)
