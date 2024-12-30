import muspy
import numpy as np
import tensorflow as tf

true_music = muspy.read_midi("/home/souraja/All_Data_OneAI_Souraja_Final/Image_to_Sequence/Generated_MIDI/testMidi_trans_enc3_dec3_VA_loss/tmp_true.mid")
pred_music = muspy.read_midi('/home/souraja/All_Data_OneAI_Souraja_Final/Image_to_Sequence/Generated_MIDI/testMidi_trans_enc3_dec3_VA_loss/tmp_pred.mid')
rep_true = muspy.to_note_representation(true_music)
rep_pred = muspy.to_note_representation(pred_music)
rep_true = np.pad(rep_true, ((0, 500 - rep_true.shape[0]), (0, 0)), 'constant', constant_values=1)
rep_pred = np.pad(rep_pred, ((0, 500 - rep_pred.shape[0]), (0, 0)), 'constant', constant_values=1)
rep_true = rep_true.reshape(1, 2000)
rep_pred = rep_pred.reshape(1, 2000)

filepath = '/home/souraja/All_Data_OneAI_Souraja_Final/Image_to_Sequence/musicupdated.keras'
model = tf.keras.models.load_model(filepath)
VA_true = model(rep_true)
VA_pred = model(rep_pred)
print(VA_true)
print(VA_pred)
new_loss = tf.reduce_mean(tf.abs(VA_true - VA_pred))
print(0.0000001*new_loss)

