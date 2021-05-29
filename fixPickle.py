import pickle

print("loading pickle")
fixPickle = pickle.load(open("data/train_doc_trf.p", "rb"))
print("saving pickle")
pickle.dump(train_doc_trf_loaded, open("data_w/train_doc_trf.p", "wb"))
