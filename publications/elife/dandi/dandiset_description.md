# Draft of the Dandiset page (FOF dataset)

Text for the fields of the Dandiset metadata form. Nothing here is uploaded by
the scripts; the fields are filled in by hand when the Dandiset is created.

## Title

One-photon calcium imaging of dorsal CA1 with synchronized behaviour in mice
freely exploring a circular open field

## Description

Calcium imaging of dorsal hippocampal CA1 in 13 adult C57BL/6J mice (5 female, 8 male) freely exploring a circular open-field arena (96 cm in diameter) with visual cues on the walls. Each mouse was recorded once a day on three consecutive days, about 12 minutes per session: 39 sessions, 26-695 neurons per session, 14,533 neuron-sessions in total. Neurons expressed GCaMP6s (AAV/DJ-CAG-GCaMP6s) and were imaged through a GRIN lens with a Miniscope V4.4; a camera above the arena filmed the animal.

Each NWB file is one session. It holds dF/F traces of the neurons kept after motion correction (NoRMCorre), source extraction (CNMF-E) and visual inspection, and behavioural variables on the time base of the imaging frames (nominal 30 frames per second): position, speed, body and head direction, rest, walk and run, freezing, rearing, and the zone of the arena the animal is in.

Not included: raw videos and the spatial footprints of the neurons.

The dataset accompanies the DRIADA paper. Scripts that reproduce its figures from these files are in the DRIADA repository (publications/elife).

## Keywords

hippocampus; CA1; calcium imaging; one-photon miniscope; GCaMP6s; open field;
spatial navigation; place cells; head direction; behavior

## Contributors

Author list and affiliations of the DRIADA paper, in its order.

| Name | Affiliations |
|---|---|
| Nikita Pospelov (corresponding author) | 1, 2 |
| Viktor Plusnin | 1, 2 |
| Olga Rogozhnikova | 1, 2 |
| Anna Ivanova | 1, 2 |
| Vladimir Sotskov | 3 |
| Margarita Orobets | 1, 4 |
| Ksenia Toropova | 1, 2 |
| Olga Ivashkina | 1, 2 |
| Vladik Avetisov | 5 |
| Konstantin Anokhin | 1, 2 |

1. Institute for Advanced Brain Studies, Lomonosov Moscow State University, Moscow, Russia
2. Moscow Physics and Technology Institute, Dolgoprudny, Russia
3. Center for Interdisciplinary Research in Biology, College de France, Paris, France
4. A.P. Nelyubin Institute of Pharmacy, I.M. Sechenov First Moscow State Medical University, Moscow, Russia
5. Semenov Institute of Chemical Physics, Russian Academy of Sciences, Moscow, Russia

## Related publication

Pospelov N, Plusnin V, Rogozhnikova O, Ivanova A, Sotskov V, Orobets M,
Toropova K, Ivashkina O, Avetisov V, Anokhin K. DRIADA: an open-source Python
toolkit for linking single-neuron selectivity and population activity to
behavior. Journal, year and DOI: to be added on publication.

Relation: the dataset is described and analysed in this publication.

## Related resources

- DRIADA, the analysis library: https://github.com/iabs-neuro/driada
- BEARMIND, the pipeline that produced the traces: https://github.com/iabs-neuro/bearmind
- SPHYNX, the pipeline that produced the behavioural variables: https://github.com/iabs-neuro/sphynx

## Ethics

All procedures were approved by the Commission of Bioethics of Lomonosov Moscow
State University (Application No. 159-g, approved during Bioethics Commission
meeting No. 154-d-r held on 17.08.2023), and followed the Russian Federation
Order N 267 MZ and the NIH Guide for the Care and Use of Laboratory Animals.

## Funding

Non-Commercial Foundation for Support of Science and Education "INTELLECT".

## To decide when the Dandiset is created

- Licence (the archive offers CC-BY-4.0 and CC0-1.0).
- Roles of the contributors and their ORCID identifiers.
- Whether the contributor list should follow this paper or the data descriptor
  in preparation, whose author list is different.
