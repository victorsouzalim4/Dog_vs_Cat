from Utils.treat_data import treat_data_set
from Utils.split_data_set import split_dataset
from CNN.generate_model import gen_model

path = "archive\PetImages"
new_directory = "archive\splitted_data"

split_dataset(path, new_directory)

#train_generator = treat_data_set(path)
#model = gen_model()