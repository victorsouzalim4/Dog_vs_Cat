from Utils.treat_data import treat_data_set
from CNN.generate_model import gen_model

path = "archive\PetImages"

train_generator = treat_data_set(path)
model = gen_model()