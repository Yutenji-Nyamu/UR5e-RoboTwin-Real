from __future__ import annotations

from pathlib import Path


def preview_file(path: Path, camera: str = "right_camera", frames: int = 4, show: bool = False) -> None:
    import h5py
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        print(f"file: {path}")
        print(f"schema_version: {handle.attrs.get('schema_version', 'legacy')}")

        def visit(name, value):
            if isinstance(value, h5py.Dataset):
                print(f"{name}: shape={value.shape}, dtype={value.dtype}")

        handle.visititems(visit)
        if not show:
            return
        if f"observation/{camera}/rgb" not in handle:
            raise KeyError(f"camera not found: {camera}")

        import cv2
        import matplotlib.pyplot as plt

        dataset = handle[f"observation/{camera}/rgb"]
        count = min(max(frames, 1), len(dataset))
        figure, axes = plt.subplots(1, count, squeeze=False, figsize=(4 * count, 4))
        for index in range(count):
            image = cv2.imdecode(np.frombuffer(dataset[index], dtype=np.uint8), cv2.IMREAD_COLOR)
            axes[0, index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, index].set_title(f"t={index}")
            axes[0, index].axis("off")
        figure.tight_layout()
        plt.show()
