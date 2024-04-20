import tensorflow as tf
####converts .keras model to saved model format. this will probably not be needed in a couple versions

#https://keras.io/guides/migrating_to_keras_3/#saving-a-model-in-the-tf-savedmodel-format
#not sure if saved model will be deprecated or not, but needs special way of saving 
#to save it as a savedmodel for tf serving
model = tf.keras.models.load_model('../neuralnetwork/dnn_model.keras')
model.export('./dnn_model_realdata')