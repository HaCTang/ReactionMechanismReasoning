from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions

rxn_smiles = "[CH2:1]=[CH:2][c:3]1[c:4]([O:5][Si:6]([CH3:7])([CH3:8])[C:9]([CH3:10])([CH3:11])[CH3:12])[cH:13][cH:14][cH:15][c:16]1[CH:17]=[C:19]1[CH2:20][CH2:21]1.[OH:18][O:24][C:23](=[O:22])[c:25]1[cH:26][cH:27][cH:28][c:29]([Cl:30])[cH:31]1>>[CH2:1]=[CH:2][c:3]1[c:4]([O:5][Si:6]([CH3:7])([CH3:8])[C:9]([CH3:10])([CH3:11])[CH3:12])[cH:13][cH:14][cH:15][c:16]1[CH:17]1[O:18][C:19]12[CH2:20][CH2:21]2.[O:22]=[C:23]([OH:24])[c:25]1[cH:26][cH:27][cH:28][c:29]([Cl:30])[cH:31]1"

rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)

img = Draw.ReactionToImage(
    rxn,
    subImgSize=(600, 300)
)

img.save("mapped_rxn.png")
