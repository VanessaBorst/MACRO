import pickle

with open('Record_names_batch1_init_epoch1.p', 'rb') as file:
    init_epoch1 = pickle.load(file)

with open('Record_names_batch1_init_epoch2.p', 'rb') as file:
    init_epoch2 = pickle.load(file)

with open('Record_names_batch1_init_epoch3.p', 'rb') as file:
    init_epoch3 = pickle.load(file)

with open('Record_names_batch1_init_epoch4.p', 'rb') as file:
    init_epoch4 = pickle.load(file)

with open('Record_names_batch1_resume_epoch3.p', 'rb') as file:
    resume_epoch3 = pickle.load(file)

with open('Record_names_batch1_resume_epoch4.p', 'rb') as file:
    resume_epoch4 = pickle.load(file)

with open('Record_names_batch1_resume_epoch4.p', 'rb') as file:
    resume_epoch4 = pickle.load(file)

with open('Record_names_batch1_resume_noRNGReload_epoch3.p', 'rb') as file:
    resume_epoch3_noRNGReload = pickle.load(file)


assert init_epoch3 == resume_epoch3 != resume_epoch3_noRNGReload
assert init_epoch4 == resume_epoch4

print("Everythings fine")