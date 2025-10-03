from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

for mol in tqdm(supplier):

    if mol is None:
        print("Skipped: unreadable molecule")
        continue

    try:
        if not mol.HasProp(target_sequence_key):
          print("Skipped: no UniProt ID")
          continue

        if not mol.HasProp(affinity_key):
            print("Skipped: missing Ki")
            continue

        ki_nM_str = mol.GetProp(affinity_key).strip()
        if not ki_nM_str or ki_nM_str.lower() == "n/a":
            print("Skipped: Ki value is n/a or empty")
            continue

        ki_nM = float(ki_nM_str)
        if ki_nM <= 0:
            print("Skipped: invalid Ki value:", ki_nM)
            continue

        # Try extracting everything without adding to records
        sequence = mol.GetProp(target_sequence_key).replace(" ", "").replace("\n", "")
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        pKi = -torch.log10(torch.tensor(ki_nM * 1e-9)).item()

        print("✅ Parsed one valid entry")

        records.append({
            "z": atomic_numbers,
            "pos": coords.tolist(),
            "seq": sequence,
            "pKi": pKi
        })
    except Exception as e:
            print("Error while parsing molecule:", e)
            continue
