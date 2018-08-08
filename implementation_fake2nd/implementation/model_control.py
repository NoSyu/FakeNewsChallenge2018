import pickle

def save_model(path, model):
    print('Save .pkl', path)
    pickle.dump(model, open(path, 'wb'))
    print('Saving finish!')



def load_model(path):
    print('Load .pkl : ', path)
    load_model = pickle.load(open(path, 'rb'))
    print('Loading finish!')
    return load_model