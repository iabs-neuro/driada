"""Create the FOF Dandiset on the DANDI archive and write dandiset.yaml.

The API key is read from the file given by --key (never printed). If a Dandiset
with the same title already belongs to the key's owner, it is reused instead of
creating a second one.

    python create_dandiset.py --key <file> --nwb-dir <folder with sub-*/>
"""
import argparse
from pathlib import Path

from dandi.dandiapi import DandiAPIClient

TITLE = ("One-photon calcium imaging of dorsal CA1 with synchronized behaviour "
         "in mice freely exploring a circular open field")

DESCRIPTION = 'Calcium imaging of dorsal hippocampal CA1 in 13 adult C57BL/6J mice (5 female, 8 male) freely exploring a circular open-field arena (96 cm in diameter) with visual cues on the walls. Each mouse was recorded once a day on three consecutive days, about 12 minutes per session: 39 sessions, 26-695 neurons per session, 14,533 neuron-sessions in total. Neurons expressed GCaMP6s (AAV/DJ-CAG-GCaMP6s) and were imaged through a GRIN lens with a Miniscope V4.4; a camera above the arena filmed the animal.\n\nEach NWB file is one session. It holds dF/F traces of the neurons kept after motion correction (NoRMCorre), source extraction (CNMF-E) and visual inspection, and behavioural variables on the time base of the imaging frames (nominal 30 frames per second): position, speed, body and head direction, rest, walk and run, freezing, rearing, and the zone of the arena the animal is in.\n\nNot included: raw videos and the spatial footprints of the neurons.\n\nThe dataset accompanies the DRIADA paper. Scripts that reproduce its figures from these files are in the DRIADA repository (publications/elife).'

KEYWORDS = ["hippocampus", "CA1", "calcium imaging", "one-photon miniscope",
            "GCaMP6s", "open field", "spatial navigation", "place cells",
            "head direction", "behavior"]

# (family, given, affiliation numbers); order and affiliations as in the paper
AUTHORS = [
    ("Pospelov", "Nikita", [1, 2]),
    ("Plusnin", "Viktor", [1, 2]),
    ("Rogozhnikova", "Olga", [1, 2]),
    ("Ivanova", "Anna", [1, 2]),
    ("Sotskov", "Vladimir", [3]),
    ("Orobets", "Margarita", [1, 4]),
    ("Toropova", "Ksenia", [1, 2]),
    ("Ivashkina", "Olga", [1, 2]),
    ("Avetisov", "Vladik", [5]),
    ("Anokhin", "Konstantin", [1, 2]),
]
AFFILIATIONS = {
    1: "Institute for Advanced Brain Studies, Lomonosov Moscow State University, Moscow, Russia",
    2: "Moscow Physics and Technology Institute, Dolgoprudny, Russia",
    3: "Center for Interdisciplinary Research in Biology, College de France, Paris, France",
    4: "A.P. Nelyubin Institute of Pharmacy, I.M. Sechenov First Moscow State Medical University, Moscow, Russia",
    5: "Semenov Institute of Chemical Physics, Russian Academy of Sciences, Moscow, Russia",
}


def metadata():
    contributors = []
    for i, (family, given, affs) in enumerate(AUTHORS):
        roles = ["dcite:Author"]
        if i == 0:
            roles.append("dcite:ContactPerson")
        contributors.append({
            "schemaKey": "Person",
            "name": f"{family}, {given}",
            "roleName": roles,
            "includeInCitation": True,
            "affiliation": [{"schemaKey": "Affiliation", "name": AFFILIATIONS[a]}
                            for a in affs],
        })
    contributors.append({
        "schemaKey": "Organization",
        "name": 'Non-Commercial Foundation for Support of Science and Education "INTELLECT"',
        "roleName": ["dcite:Funder"],
        "includeInCitation": False,
    })
    return {
        "name": TITLE,
        "description": DESCRIPTION,
        "license": ["spdx:CC-BY-4.0"],
        "keywords": KEYWORDS,
        "contributor": contributors,
        "ethicsApproval": [{
            "schemaKey": "EthicsApproval",
            "identifier": ("Commission of Bioethics of Lomonosov Moscow State University, "
                           "Application No. 159-g, meeting No. 154-d-r, 17.08.2023"),
            "contactPoint": {"schemaKey": "ContactPoint"},
        }],
        "relatedResource": [
            {"schemaKey": "Resource", "url": "https://github.com/iabs-neuro/driada",
             "name": "DRIADA", "relation": "dcite:IsSupplementedBy"},
            {"schemaKey": "Resource", "url": "https://github.com/iabs-neuro/bearmind",
             "name": "BEARMIND", "relation": "dcite:IsCompiledBy"},
            {"schemaKey": "Resource", "url": "https://github.com/iabs-neuro/sphynx",
             "name": "SPHYNX", "relation": "dcite:IsCompiledBy"},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--nwb-dir", required=True)
    ap.add_argument("--instance", default="dandi")
    args = ap.parse_args()

    token = Path(args.key).read_text().strip()
    client = DandiAPIClient.for_dandi_instance(args.instance, token=token)
    me = client.get("/users/me/")
    print("[OK] key accepted, user:", me.get("username"))

    mine = client.get("/dandisets/", params={"user": "me", "draft": "true",
                                             "empty": "true", "embargoed": "true"})
    existing = [client.get_dandiset(r["identifier"]) for r in mine["results"]
                if (r.get("draft_version") or {}).get("name") == TITLE]
    if existing:
        ds = existing[0]
        print("[OK] reusing existing Dandiset", ds.identifier)
    else:
        ds = client.create_dandiset(TITLE, metadata())
        print("[OK] created Dandiset", ds.identifier)

    yaml_path = Path(args.nwb_dir) / "dandiset.yaml"
    yaml_path.write_text(f"identifier: '{ds.identifier}'\nname: '{TITLE}'\n",
                         encoding="utf-8")
    print("[OK] wrote", yaml_path)
    print("URL:", f"{client.dandi_instance.gui}/dandiset/{ds.identifier}/draft")


if __name__ == "__main__":
    main()
