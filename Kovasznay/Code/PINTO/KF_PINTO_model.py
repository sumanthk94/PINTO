import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
cdir = os.getcwd()
os.chdir(cdir)
import tensorflow as tf
from tensorflow import keras
from keras import layers
import wandb
import numpy as np
import pandas as pd
from KF_PINTO_Pde import PdeModel
from utils import get_ibc_and_inner_data

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
grid_size = 64
train_nue = [1 / 20, 1 / 30, 1 / 50, 1/80]
seq_len = 20
domain_samples = 2000

(xd, yd, xb, yb, ub, vb, pb,
 xbc_in, ybc_in, ubc_in, vbc_in, pbc_in,
 xbc_b, ybc_b, ubc_b, vbc_b, pbc_b, nue_d, sen_coord) = get_ibc_and_inner_data(start=[-0.5, -0.5],
                                                                               stop=[1., 1.5],
                                                                               grid_size=grid_size,
                                                                               domain_samples=domain_samples,
                                                                               seq_len=seq_len,
                                                                               nue=train_nue)

ivals = {'xin': xd, 'yin': yd, 'xb': xb, 'yb': yb,
         'xbc_in': xbc_in, 'ybc_in': ybc_in, 'ubc_in': ubc_in, 'vbc_in': vbc_in, 'pbc_in': pbc_in,
         'xbc_b': xbc_b, 'ybc_b': ybc_b, 'ubc_b': ubc_b, 'vbc_b': vbc_b, 'pbc_b': pbc_b,
         'nue_d': nue_d}
ovals = {'ub': ub, 'vb': vb, 'pb': pb}
parameters = {'train_nue': train_nue, 'test_nue': [1 / 10, 1 / 20, 1 / 30, 1 / 40, 1 / 50]}

# Building PINTO model using Functional API
initializer = tf.keras.initializers.GlorotUniform(1234)


# Define the sequential model for query value transfer
def get_model(model_name, layer_names, layer_units, activation='swish'):
    sq = keras.Sequential(name=model_name)
    for i in range(len(layer_units)):
        sq.add(layers.Dense(units=layer_units[i], kernel_initializer=initializer,
                            name=layer_names + str(i + 1)))
        sq.add(layers.Activation(activation=activation))
    return sq


# Lifting operation for query points
input1 = keras.Input(shape=(1,), name='X_layer')
rescale_input1 = layers.Rescaling(scale=2 / 1.5, offset=-0.5 / 1.5)(input1)
input2 = keras.Input(shape=(1,), name='Y_layer')
rescale_input2 = layers.Rescaling(scale=1, offset=-0.5)(input2)
sp_trans = get_model(model_name='spatial_transformation',
                     layer_names='spatial_layer',
                     layer_units=[64, 64], activation='swish')

sp = layers.Concatenate()([rescale_input1, rescale_input2])
sp = layers.Reshape(target_shape=(1, -1))(sp)
spq = sp_trans(sp)
residual = spq

#  MLP for BPE unit
input3 = keras.Input(shape=(None, 1,), name='Xbc_layer')
rescale_input3 = layers.Rescaling(scale=2 / 1.5, offset=-0.5 / 1.5)(input3)
input4 = keras.Input(shape=(None, 1,), name='Ybc_layer')
rescale_input4 = layers.Rescaling(scale=1, offset=-0.5)(input4)

pe = layers.Concatenate()([rescale_input3, rescale_input4])
pe = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(pe)
pe = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(pe)

# MLP for BVE unit
input5 = keras.Input(shape=(None, 1,), name='Ubc_layer')
input6 = keras.Input(shape=(None, 1,), name='Vbc_layer')
input7 = keras.Input(shape=(None, 1,), name='Pbc_layer')

ce = layers.Concatenate()([input5, input6, input7])
ce = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(ce)
ce = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(ce)

# cross-attention layer
ct = layers.MultiHeadAttention(num_heads=2, key_dim=64)(query=spq, key=pe, value=ce)
ct = layers.Add()([residual, ct])
ct = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(ct)
ct = layers.Flatten()(ct)

# projection operators for output functional space

# output model for u
ou = get_model(model_name='U', layer_units=[64, 64], layer_names='ou',
               activation='swish')(ct)
ou = layers.Dense(units=1, kernel_initializer=initializer, name='output_u')(ou)

# output model for v
ov = get_model(model_name='V', layer_units=[64, 64], layer_names='ov', activation='swish')(ct)
ov = layers.Dense(units=1, kernel_initializer=initializer, name='output_v')(ov)

# output model for p
op = get_model(model_name='P', layer_units=[64, 64], layer_names='op', activation='swish')(ct)
op = layers.Dense(units=1, kernel_initializer=initializer, name='output_p')(op)

# building the PINTO model
model = keras.Model([input1, input2, input3, input4, input5, input6, input7],
                    [ou, ov, op])

model.summary()

# training the model
initial_learning_rate = 5e-4

## Uncomment the following lines to use a learning rate scheduler
## Exponential decay learning rate scheduler

# decay_steps = 10000
# decay_rate = 0.5
# staircase = True
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     decay_rate=decay_rate,
#     staircase=staircase)

# initiating optimizer and loss function
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
model_dict = {"nn_model": model}

# metrics to track the model performance during training
metrics = {"loss": keras.metrics.Mean(name='loss'),
           "boundary_loss": keras.metrics.Mean(name='std_loss'),
           "residual_loss": keras.metrics.Mean(name='residual_loss'),
           "u_loss": keras.metrics.Mean(name='u_loss'),
           "v_loss": keras.metrics.Mean(name='v_loss'),
           "p_loss": keras.metrics.Mean(name='p_loss')
           }
batches = 5

# initiating the PdeModel class
cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, batches=batches)

epochs = 40000
vf = 100  # verbose frequency
pf = 1000  # plot frequency
wb = False  # wandb logging

configuration = {'#_total_initial_and_boundary_points': len(xb),
                 '#_total_domain_points': len(xd),
                 "optimizer": 'Adam',
                 "batches": batches,
                 "learning_rate": initial_learning_rate,
                 # "initial_lr": 'NA',
                 # "decay_rate": 'NA',
                 # "decay_steps": 'NA',
                 # "staircase": 'NA',
                 # "lr_scheduler": 'NA',
                 # 'step_lim': 'NA',
                 "Epochs": epochs,
                 "Activation": 'swish',
                 "model_name": 'KF_PINTO_model',
                 "trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.trainable_weights]),
                 "non_trainable_parameters": np.sum([np.prod(lay.shape) for lay in model.non_trainable_weights]),
                 "nue": parameters['test_nue'],
                 "kernel_initializer": 'GlorotUniform',
                 "sequence_length": 4 * seq_len}

print(configuration)

if wb:
    wandb.init(project='Finalising_results', config=configuration)

log_dir = 'output/KF_PINTO/'
try:
    os.makedirs(log_dir)
except FileExistsError:
    pass

history = cm.run(epochs=epochs, log_dir=log_dir, wb=wb, verbose_freq=vf, plot_freq=pf)
pd.DataFrame(sen_coord).to_csv(log_dir + 'sensor_data.csv')

if wb:
    wandb.finish()

# Evaluation
cm.nn_model.save(log_dir + 'KF_PINTO_model', save_format='tf')
