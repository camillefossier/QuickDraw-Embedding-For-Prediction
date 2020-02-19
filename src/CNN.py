import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

# Supposons on a en entré (pour chaque dessin) une matrice de taille n*p soit 120*5

# Construction du modèle
num_filters = 10 
filter_size = (3,3) # Taille de fenêtre = 3*3
stride = (1,1) # La fenêtre se déplace de 1 à l'horizontal et 1 à la vertical
input_shape = (120,5) 
pool_size = (2,2) # factors by which to downscale (vertical, horizontal)
num_classes = 3 # Nombre de classes

model = Sequential()
model.add(Conv2D(filters = num_filters, kernel_size = filter_size, strides=stride,
                 activation = 'relu',
                 input_shape = input_shape))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Compilation du modèle
model.compile(
  optimizer = 'adam',                #  Adam gradient-based optimizer (Possibilité de changer)
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Entrainement du modèle
batch_size = 50
epochs = 15

# /!\ Pour le y, possibilité d'utiliser la fonction to_categorical /!\
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 1,                # 1 = Progress Bar visible
          validation_data=(x_test, y_test))

# Sauvegarder le model
model.save_weights('cnn.h5')
# Charger le model
model.load_weights('cnn.h5')


