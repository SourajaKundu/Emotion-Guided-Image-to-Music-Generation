from experiments import valence, arousal
import tensorflow as tf
import muspy
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

batch_size = 1

class MyDataset:
    def __init__(self, midis_path):
        self.midis_path = midis_path
        self.midis = os.listdir(midis_path)
        self.midis.sort()
        self.num_samples = len(self.midis)

    def __len__(self):
        return self.num_samples // batch_size

    def __call__(self):
        for index in range(len(self)):
            reps = []
            vas = []
            for i in range(batch_size * index, min(batch_size * (index + 1), self.num_samples)):
                midi = self.midis[i]
                muse_path = os.path.join(self.midis_path, midi)
                
                music = muspy.read_midi(muse_path)
                rep = muspy.to_note_representation(music)
                rep = np.pad(rep, ((0, 500 - rep.shape[0]), (0, 0)), 'constant', constant_values=1)
                rep = rep.reshape(2000)  # Flatten the representation to match the input dimension
                reps.append(rep)
                
                v = valence[i]
                a = arousal[i]
                va = [v, a]
                vas.append(va)
            
            reps = np.array(reps, dtype=np.float32)
            vas = np.array(vas, dtype=np.float32)
            yield reps, vas


midis_path = '/home/souraja/All_Data_OneAI_Souraja_Final/Image_to_Sequence/train_data/audio'


# Define the dataset (as in your code)
dataset = MyDataset(midis_path)
dataset = tf.data.Dataset.from_generator(
    dataset,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, 2000), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, 2), dtype=tf.float32)
    )
)

# Learning rate schedule
initial_learning_rate = 0.0001
decay_steps = 250
decay_rate = 0.1

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=False
)

# Compile the model with the RMSprop optimizer using the learning rate schedule
optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

# Define the model
model = tf.keras.Sequential([
    layers.Dense(1024, input_dim=2000, use_bias=False, activation='relu'),
    layers.Dense(512, use_bias=False, activation='relu'),
    layers.Dense(2, use_bias=False)
])

# Compile the model
model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0000001), loss='mse')

# Define the checkpoint directory and callback
checkpoint_dir = "/home/souraja/All_Data_OneAI_Souraja_Final/Image_to_Sequence/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-{epoch:04d}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,  # Only save the model's weights
    save_freq='epoch',       # Save at the end of every epoch
    verbose=1
)


# Find the latest weights file
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".weights.h5")]
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('-')[1].split('.')[0]))
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading weights from: {latest_checkpoint_path}")
    model.load_weights(latest_checkpoint_path)
else:
    print("No checkpoint found, starting training from scratch.")

# Train the model with the checkpoint callback
model.fit(dataset, epochs=10, callbacks=[checkpoint_callback])

# Save the final model in .keras format after training
final_model_path = '/home/souraja/All_Data_OneAI_Souraja_Final/Image_to_Sequence/musicupdated.keras'
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

