import os
import shutil
import random
from pathlib import Path

def split_dataset(base_dir, output_dir, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """
    Separa as imagens em train, val e test em novas pastas.
    
    base_dir: caminho para a pasta com as classes (ex: cats, dogs)
    output_dir: onde criar as novas pastas (train/, val/, test/)
    split_ratio: tupla com proporção (train, val, test)
    """
    assert sum(split_ratio) == 1.0, "A soma dos splits deve ser 1.0"
    random.seed(seed)

    classes = [d.name for d in Path(base_dir).iterdir() if d.is_dir()]

    for cls in classes:
        cls_path = Path(base_dir) / cls
        images = list(cls_path.glob("*.*"))
        random.shuffle(images)

        total = len(images)
        train_end = int(split_ratio[0] * total)
        val_end = train_end + int(split_ratio[1] * total)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        for split_name, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            split_dir = Path(output_dir) / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for file in file_list:
                shutil.copy2(file, split_dir / file.name)

    print("✅ Separação concluída com sucesso!")