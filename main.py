from Utils.treat_data import treat_data_set
from Utils.split_data_set import split_dataset
from CNN.generate_model import gen_model

train_path = "archive/splitted_data/train"
val_path = "archive/splitted_data/val"
test_path = "archive/splitted_data/test"

train_generator = treat_data_set(train_path)
val_generator = treat_data_set(val_path)
test_generator = treat_data_set(test_path)

model = gen_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")