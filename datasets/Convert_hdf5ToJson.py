import h5py
import json
import numpy as np

def hdf5_to_dict(h5obj):
    """
    Recursively convert h5py Group/Dataset to pure python dict/list
    """
    if isinstance(h5obj, h5py.Dataset):
        data = h5obj[()]
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.bytes_, bytes)):
            return data.decode("utf-8", errors="ignore")
        else:
            return data.item() if hasattr(data, "item") else data

    elif isinstance(h5obj, h5py.Group):
        out = {}

        # read attributes
        if len(h5obj.attrs) > 0:
            out["_attrs"] = {}
            for k, v in h5obj.attrs.items():
                if isinstance(v, np.ndarray):
                    out["_attrs"][k] = v.tolist()
                else:
                    out["_attrs"][k] = v.item() if hasattr(v, "item") else v

        # read children
        for key in h5obj.keys():
            out[key] = hdf5_to_dict(h5obj[key])

        return out


if __name__ == "__main__":
    file_name = "dataset"  # without the extension
    input_h5 = f"{file_name}.hdf5"
    output_json = f"{file_name}.json"

    with h5py.File(input_h5, "r") as f:
        data_dict = hdf5_to_dict(f)

    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(data_dict, fp, indent=2, ensure_ascii=False)

    print(f"✅ Converted: {input_h5} → {output_json}")
