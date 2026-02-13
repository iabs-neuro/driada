"""Example: load NOF data and explore the NeuronDatabase."""

import sys
from pathlib import Path

# Add tools/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuron_database import NeuronDatabase, load_from_csv_directory


def main():
    data_dir = Path(__file__).parent.parent.parent / "DRIADA data" / "NOF"

    matching, data = load_from_csv_directory(
        data_dir, session_names=['1D', '2D', '3D', '4D']
    )
    db = NeuronDatabase(['1D', '2D', '3D', '4D'], matching, data)
    db.summary()

    # Direct pandas access
    df = db.data
    print(f"\nAll features: {db.features}")
    print(f"All mice: {db.mice}")

    # Significant place cells with MI > 0.04
    place_sig = db.query(feature='place', significant=True, mi_min=0.04)
    print(f"\nSignificant place cells (MI>0.04): {len(place_sig)} entries")
    print(place_sig[['mouse', 'session', 'matched_id', 'me']].head(10))

    # Neuron count per mouse × session
    print(f"\nNeurons per mouse x session:")
    print(db.n_neurons())


if __name__ == '__main__':
    main()
