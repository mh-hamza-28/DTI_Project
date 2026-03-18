try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError:
    pass

def drug_features(smiles):
    """
    Extracts features from SMILES using RDKit:
    1. Molecular weight
    2. Number of hydrogen donors
    3. Number of hydrogen acceptors
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mw = Descriptors.MolWt(mol)
        hdonors = rdMolDescriptors.CalcNumHBD(mol)
        hacceptors = rdMolDescriptors.CalcNumHBA(mol)
        return mw, hdonors, hacceptors
    except Exception:
        # Graceful handling for invalid SMILES as requested
        return None

def protein_features(sequence):
    """
    Extracts protein features depending on sequence:
    1. Count of A
    2. Count of L
    3. Count of G
    4. Total length
    """
    try:
        if not sequence or not isinstance(sequence, str):
            return None
        seq = sequence.upper()
        a_count = seq.count('A')
        l_count = seq.count('L')
        g_count = seq.count('G')
        length = len(seq)
        return a_count, l_count, g_count, length
    except Exception:
        return None
