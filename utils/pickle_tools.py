import pickle


def load_pickle(target_name):
    print 'Loading pickle:', target_name, '...'
    with open(target_name, 'rb') as f:
        target_data = pickle.load(f)
        
    return target_data


def write_pickle(to_save_name, target_data):
    print 'Saving pickle:', to_save_name, '...'
    with open(to_save_name, 'wb') as f:
        pickle.dump(target_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
