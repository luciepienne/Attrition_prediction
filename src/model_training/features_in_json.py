import json

import numpy as np


def save_feature_info(
    feature_names, feature_types, encoding_dict, output_file="models/feature_info.json"
):
    """Sauvegarder les informations sur les features dans un fichier JSON."""


    feature_info = {
        "feature_names": feature_names,
        "feature_types": {name: str(dtype) for name, dtype in feature_types.items()},
        "encoding_dict": {
            key: {str(k): int(v) for k, v in value.items()}
            for key, value in encoding_dict.items()
        },
    }

    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


    with open(output_file, "w") as f:
        json.dump(feature_info, f, indent=2, default=convert_to_json_serializable)

    print(f"Informations sur les features sauvegardées dans {output_file}.")
