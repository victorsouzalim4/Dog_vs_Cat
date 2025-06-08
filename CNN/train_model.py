from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def train_model(model, train_generator, val_generator):
        
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    os.makedirs("Model", exist_ok=True)


    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath="Model/melhor_modelo.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]


    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=callbacks
    )


    model.save("Model/dog_vs_cat_model_final.keras")
    print("✅ Modelo final salvo em: Model/dog_vs_cat_model_final.keras")