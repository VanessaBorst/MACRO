import pickle

with open('Record_names_train_init.p', 'rb') as file:
    train_init = pickle.load(file)

with open('Record_names_valid_init.p', 'rb') as file:
    valid_init = pickle.load(file)

with open('Record_names_train_resume.p', 'rb') as file:
    train_resume = pickle.load(file)

with open('Record_names_valid_resume.p', 'rb') as file:
    valid_resume = pickle.load(file)

assert train_init == train_resume
assert valid_init == valid_resume

print("Everythings fine")