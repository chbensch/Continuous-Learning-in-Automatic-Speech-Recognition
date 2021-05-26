import pickle

with open('data/val_doc_trf.pickle', 'rb') as handle:
    val_doc_trf_loaded = pickle.load(handle)

print(len(val_doc_trf_loaded))