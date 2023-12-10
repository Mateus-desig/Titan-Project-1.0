import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower())
    outputs.append('{}\{}'.format(command['entity'], command['action']))

# Processar texto: palavras, caracteres, bytes, sub-palavras.


max_seq = max([len(bytes(x.encode('utf-8'))) for x in inputs])

print(max_seq)


# Criar dataset (números de exemplos, tamanho da seq, num caracteres) one-hot.
# Criar dataset desperso (números de exemplos, tamanho da seq).

# Input data one-hot encoding.

input_data = np.zeros((len(inputs), max_seq, 256), dtype='float32')

for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i ,k , int(ch)] = 1.0

# Input data sparse.
# input_data = np.zeros((len(inputs), max_seq), dtype='int32')

# for i, input in enumerate(inputs):
#     for k, ch in enumerate(input):
#         input_data[i ,k] == chr2idx[ch]

# Output Data.

labels = set(outputs)

fwrite = open('labels.txt', 'w', encoding="utf-8")

labels2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    labels2idx[label] = k
    idx2label[k] = label
    fwrite.write(label + '\n')
fwrite.close()

output_data = []

for output in outputs:
    output_data.append(labels2idx[output])

output_data = to_categorical(output_data, len(output_data))


print(output_data[0])

model = Sequential()
model.add(LSTM(128))
model.add(Dense(len(output_data), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(input_data, output_data, epochs=128)

# Salvar modelo.
model.save('model.h5')

# Classificar texto em uma entidade.
def classify(text):
    # Criar um arrey de entrada.
    x = np.zeros((1, 48, 256), dtype='float32')

    # Prencher o arrey com dados do texto.
    for k, ch in enumerate(bytes(text.encode('utf-8'))):
        x[0, k, int(ch)] = 1.0

    # Fazer a previsão.
    out = model.predict(x)
    idx = out.argmax()
    print(idx2label[idx])

'''while True:
    text = input("Digite algo: ")
    classify(text)'''