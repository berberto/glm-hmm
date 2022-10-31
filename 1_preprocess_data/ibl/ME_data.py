from one_global import one

import pandas as pd
from pprint import pprint
import pickle
import numpy as np


if __name__ == "__main__":

	ibl_data_path = "../../data/ibl"


	df = pd.read_csv(f"{ibl_data_path}/ME.csv")#, index_col=True)

	eids_ME = df['eid'].to_numpy()
	print(eids_ME)

	animals_ME = []
	for eid in eids_ME:
		# print("\n=======================================")
		# print(eid)
		# print("---------------------------------------")
		animal = str(one.eid2path(eid)).split("/")[-3]
		animals_ME.append(animal)

	subjects = list(np.unique(np.array(animals_ME)))
	eids, info = [], []
	for subject in subjects:
		_eids, _info = one.search(subject=subject, details=True)
		eids = eids + _eids
		info = info + _info
		print(f"{subject}\t{len(_eids)}")

	with open(f"{ibl_data_path}/eids_info_ME.pkl", "wb") as f:
		pickle.dump([eids,info], f)

	eids_neural, info_neural = [], []
	for _eid in eids_ME:
		idx = eids.index(eid)
		_info = info[idx]
		eids_neural.append(_eid)
		info_neural.append(_info)
		print("\n==========================")
		print(_eid)
		print("--------------------------")
		pprint(_info)

	with open(f"{ibl_data_path}/eids_info_neural_ME.pkl", "wb") as f:
		pickle.dump([eids_neural,info_neural], f)
