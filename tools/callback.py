from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class SaveBestLossModel(ModelCheckpoint):
    def __init__(self, model_name):
        super().__init__(model_name, save_best_only=True, monitor='val_loss', mode='min')

class SaveBestAccuModel(ModelCheckpoint):
    def __init__(self, model_name):
        super().__init__(model_name, save_best_only=True, monitor='val_accuracy', mode='max')

class ReduceLearningRate(ReduceLROnPlateau):
    def __init__(self):
        super().__init__(monitor="val_loss",
                         patience=4, # si val_loss stagne sur 4 epochs consécutives selon la valeur min_delta
                         min_delta=0.01,
                         factor=0.1,  # On réduit le learning rate d'un facteur 0.1
                         cooldown=4,  # On attend 4 epochs avant de réitérer 
                         verbose=1)

class CustomEarlyStopping(EarlyStopping):
    def __init__(self):
        super().__init__(patience=5, # Attendre 5 epochs avant application
                         min_delta=0.01, # si au bout de 5 epochs la fonction de perte ne varie pas de 1%, 
                         verbose=1, # Afficher à quel epoch on s'arrête
                         mode='min',
                         monitor='val_loss')
