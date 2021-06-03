from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import cv2
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1


image_model=load_model('image_model.h5')
with open("vocabulary.pkl","rb") as f:
        vocabulary=pickle.load(f)

with open("invert_vocab.pkl","rb") as f:
        invert_vocab=pickle.load(f)

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(vocabulary) + 1
#num_steps = len(x_train) // BATCH_SIZE
features_shape = 512
attention_features_shape = 49
max_length=35

class VGG16_Encoder(tf.keras.Model):
  def __init__(self, embedding_dim):
    super(VGG16_Encoder, self).__init__()
    self.fc = tf.keras.layers.Dense(embedding_dim)
    self.dropout = tf.keras.layers.Dropout(0.5)
  def call(self, x):
    x= self.dropout(x)
    x = self.fc(x)
    x = tf.nn.relu(x)
    return x

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
    score = self.V(attention_hidden_layer)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units
    self.vocab_size=vocab_size
    self.embedding_dim=embedding_dim

    self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

encoder = VGG16_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

checkpoint_path = "train/ckpt-8"
ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
ckpt.restore(checkpoint_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict():
    f=request.files['file1']

    f.save("static/image_file.jpg")
    
    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))
        hidden = decoder.reset_state(batch_size=1)
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (224,224))
        img = np.reshape(img, (1,224,224,3))
        img_tensor_val =image_model.predict(img)#.reshape(1,-1,512)
    #print(img_tensor_val.shape)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],-1,img_tensor_val.shape[3]))
        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([vocabulary['start']], 0)
        result = []
        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input,features, hidden)
            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(invert_vocab[predicted_id])

            if invert_vocab[predicted_id] == 'end':
                return result, attention_plot
            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot
    result,a=evaluate("static/image_file.jpg")
    caption=(" ").join([x for x in result])
    return render_template('predict.html',prediction_texts=caption)

    
       

if __name__=="__main__":
    app.run(debug=True)

