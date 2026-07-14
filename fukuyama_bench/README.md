# FukuyamaBench

319 organic reaction mechanisms from Fukuyama’s Advanced Organic Reaction Mechanism workbook.

| Partition | Cases |
|-----------|------:|
| A | 78 |
| B | 131 |
| C | 110 |

Each case directory contains:

- `mechanism.json` — overall reaction + elementary steps
- `rxn.json` — step SMILES (`reactants>>products`)
- `mapped_rxn.json` — atom-mapped step SMILES
- `ckpt.txt` — key intermediate checkpoints

Construction scripts: `build/`. Images are omitted from git.
