print("no problem")

from numpy import array
from pickle import load, dump
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from glob import glob
import os
from numpy import argmax
from music21 import converter, instrument, note, chord, stream, meter,duration, interval, pitch, tempo, midi
from tensorflow.keras.applications.inception_v3 import InceptionV3

from os import listdir
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, Add, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from functools import partial
import tensorflow as tf
import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


install('music21')



#data folder path
data_dir = ""
#midi note and duration separator
separator = "@"
#image classifier model labels
label_VGG16 = "VGG16"
label_InceptionV3 = "InceptionV3"
#image model currently in use
current_ImgModel = label_InceptionV3

# songs = glob(data_dir + 'midi/*.MID')
songs = glob('audio/*.mid')
songs.sort()



def create_model_with_2_transformer(vocab_size, max_length, feature_extractor_label="InceptionV3"):
    # Image feature extractor model
    if feature_extractor_label == "VG16":
        inputs1 = Input(shape=(4096,))
    elif feature_extractor_label == "InceptionV3":
        inputs1 = Input(shape=(2048,))

    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence model with Transformer
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.2)(se1)

    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(se2, se2)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.2)(attention_output)

    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(attention_output, attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.2)(attention_output)

    se3 = GlobalAveragePooling1D()(attention_output)

    # Decoder model
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(128, activation='relu')(decoder1)
    decoder3 = Dense(256, activation='relu')(decoder2)
    outputs = Dense(vocab_size, activation='softmax')(decoder3)

    # Merging model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001))

    # Print model summary
    model.summary()

    return model

def create_model_with_4_transformer_MIDI(vocab_size, max_length, feature_extractor_label="VG16"):
    # Image feature extractor model
    if feature_extractor_label == "VG16":
        inputs1 = Input(shape=(4096,))
    elif feature_extractor_label == "InceptionV3":
        inputs1 = Input(shape=(2048,))

    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence model with Transformer
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.2)(se1)

    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(se2, se2)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.2)(attention_output)

    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(attention_output, attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.2)(attention_output)

    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(attention_output, attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.2)(attention_output)

    attention_output = MultiHeadAttention(num_heads=4, key_dim=256)(attention_output, attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.2)(attention_output)

    se3 = GlobalAveragePooling1D()(attention_output)

    # Decoder model
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(128, activation='relu')(decoder1)
    decoder3 = Dense(256, activation='relu')(decoder2)
    outputs = Dense(vocab_size, activation='softmax')(decoder3)

    # Merging model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001))

    # Print model summary
    model.summary()

    return model


def listToString(listObj):

    # initialize an empty string
    stringText = ""

    # traverse in the string
    for element in listObj:
        stringText = stringText + " " + element

    # return string
    return stringText

def transpose(song):
    #Transposes song to C maj/A min

    # get key using music21
    print("Ok")
    key = song.analyze("key")
    print(key,"Ok")
    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        intervalSong = interval.Interval(key.tonic, pitch.Pitch("C"))
    elif key.mode == "minor":
        intervalSong = interval.Interval(key.tonic, pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(intervalSong)
    return tranposed_song


#function to check if rest duration is acceptables
def isRestAcceptable(rest, timeSignature):
  #check is time signature denominator is 4
  if (timeSignature.denominator == 4):
      #if denominator 4, length of one bar is 4 beats
      barDuration = timeSignature.numerator
  #check is time signature denominator is 8
  elif (timeSignature.denominator == 8):
      #if denominator is 8, length of one bar is numerator/2
      barDuration = timeSignature.numerator/2.0
  #check is time signature denominator is 2
  elif (timeSignature.denominator == 2):
      #if denominator is 2, length of one bar is numerator * 2
      barDuration = timeSignature.numerator * 2.0
  else : barDuration = 0

  #return false if rest duration is greater than 2 bars
  if barDuration * 2 < rest.quarterLength:
      return False

  return True


def getStartingBar(score):
    for i in range(51):
        bar = score.measures(i-1,i)
        barflat = bar.flatten()
        for event in barflat:
            if isinstance(event, note.Note):
                num = i
                break
        else:
            continue
        break
    return num


def createDatafile():
  #create a file named midiData.txt and append
  midiData_file = open("midiData.txt", "w")
#iterate over each midi file
  print("Writing midi data to file...")
  songCount = 0
  for file in songs:
    print("Processing " + str(file))
    print(file)
    try:
        midi = converter.parse(file) #convert to midi
    except:
        print("Error in parsing " + str(file))
        continue
    print("midi = ",midi)
    #midi = midi.stripTies()
    #tranpose song
    try:
        midi = transpose(midi)
    except:
        print("Error in obtaining key for " + str(file))
        continue
    #obtain 30 bars
    songTempo = ''
    try:
        firstbar = midi.measures(0,3)
        firstbarNotes = firstbar.flatten()
        for element in firstbarNotes:
            if isinstance(element, tempo.MetronomeMark):
                songTempo = str(element.number) + separator + "tempo"
        startBar = getStartingBar(midi)
        midi = midi.measures(startBar, startBar + 7)
    except Exception as e:
      print("Error in measures for " + str(file))
      continue
    midi = midi.flatten() #combine all parts and get a single notes part

    songCount += 1
    #initialize timeSignature, notes list and count variables
    timeSignature = ''
    notes = []
    #count = 0

    #loop over each midi event
    #midi events include timesignatures, instruments, notes, rests and chords
    for event in midi:
      #only obtain 50 events
      # if count == 50:
      #   break;
      # count += 1 #increase count
      #check if event is a timeSignature
      if isinstance(event, meter.TimeSignature ):
        #save time signature to file as '3/4@time'
        timeSignature = event.ratioString + separator + "time"
        timeSignatureEvent = event
      #check if event is a note
      if isinstance(event, note.Note):
        #save note to file as 'C1@0.5' where 'C1' is midi note and '0.5' is the duration
        notes.append(str(event.pitch) + separator + str(event.quarterLength))
      #check if event is a chord
      elif(isinstance(event, chord.Chord)):
        #save chord to file as '1.8.2@0.5' where '1.8.2' are notes in the chord and '0.5' is the duration
        notes.append(('.'.join(str(n) for n in event.normalOrder))+ separator + str(event.quarterLength))
      #check if event is a rest
      elif(isinstance(event, note.Rest)):
        #check if rest duration is acceptable
        if isRestAcceptable(event, timeSignatureEvent):
          #save rest as 'r@0.5' where 'r' indicates that it is a rest and '0.5'  is the duration
          notes.append('r' + separator + str(event.quarterLength))

    #save the sequence to the file
    # jpgFilenamesList = glob(data_dir + 'images/' + os.path.basename(file).split('.')[0] + '*.*')
    jpgFilenamesList = ['image/' + os.path.basename(file).split('.')[0] + '.jpg']
    #jpgFilenamesList = glob(
    simImageCount = 0
    for image in jpgFilenamesList:
        simImageCount += 1
        #first write the file id which is same as the image id
        sequence = os.path.basename(image).split('.')[0] + " " + "<start> " + songTempo + " " + timeSignature + listToString(notes) + " <end>" #append <start> and <end> tokens
        #if current song is the last in the list, don't add a newline character to the end
        if((len(songs)== songs.index(file) +1) and simImageCount == len(jpgFilenamesList)):
            pass
        else:
            sequence = sequence + "\n"
        midiData_file.write(sequence)

  print(f"Finishes writing midi data. {songCount} songs written.")
  #close file
  midiData_file.close()


def load_file(fileName):
  """

  Args:
    fileName:

  Returns:
    content:

  """
  #open file in read mode
  file = open(fileName, 'r')
  #obtain content in file
  content = file.read()
  #close file
  file.close()

  return content


# def createDatafile():
#   #create a file named midiData.txt and append
#   midiData_file = open("midiData.txt", "w")
# #iterate over each midi file
#   print("Writing midi data to file...")
#   songCount = 0
#   for file in songs:
#     print("Processing " + str(file))
#     print(file)
#     try:
#         midi = converter.parse(file) #convert to midi
#     except:
#         print("Error in parsing " + str(file))
#         continue
#     print("midi = ",midi)
#     #midi = midi.stripTies()
#     #tranpose song
#     try:
#         midi = transpose(midi)
#     except:
#         print("Error in obtaining key for " + str(file))
#         continue
#     #obtain 30 bars
#     songTempo = ''
#     try:
#         firstbar = midi.measures(0,3)
#         firstbarNotes = firstbar.flatten()
#         for element in firstbarNotes:
#             if isinstance(element, tempo.MetronomeMark):
#                 songTempo = str(element.number) + separator + "tempo"
#         startBar = getStartingBar(midi)
#         midi = midi.measures(startBar, startBar + 7)
#     except Exception as e:
#       print("Error in measures for " + str(file))
#       continue
#     midi = midi.flatten() #combine all parts and get a single notes part

#     songCount += 1
#     #initialize timeSignature, notes list and count variables
#     timeSignature = ''
#     notes = []
#     #count = 0

#     #loop over each midi event
#     #midi events include timesignatures, instruments, notes, rests and chords
#     for event in midi:
#       #only obtain 50 events
#       # if count == 50:
#       #   break;
#       # count += 1 #increase count
#       #check if event is a timeSignature
#       if isinstance(event, meter.TimeSignature ):
#         #save time signature to file as '3/4@time'
#         timeSignature = event.ratioString + separator + "time"
#         timeSignatureEvent = event
#       #check if event is a note
#       if isinstance(event, note.Note):
#         #save note to file as 'C1@0.5' where 'C1' is midi note and '0.5' is the duration
#         notes.append(str(event.pitch) + separator + str(event.quarterLength))
#       #check if event is a chord
#       elif(isinstance(event, chord.Chord)):
#         #save chord to file as '1.8.2@0.5' where '1.8.2' are notes in the chord and '0.5' is the duration
#         notes.append(('.'.join(str(n) for n in event.normalOrder))+ separator + str(event.quarterLength))
#       #check if event is a rest
#       elif(isinstance(event, note.Rest)):
#         #check if rest duration is acceptable
#         if isRestAcceptable(event, timeSignatureEvent):
#           #save rest as 'r@0.5' where 'r' indicates that it is a rest and '0.5'  is the duration
#           notes.append('r' + separator + str(event.quarterLength))

#     #save the sequence to the file
#     # jpgFilenamesList = glob(data_dir + 'images/' + os.path.basename(file).split('.')[0] + '*.*')
#     jpgFilenamesList = ['image/' + os.path.basename(file).split('.')[0] + '.jpg']
#     #jpgFilenamesList = glob(
#     simImageCount = 0
#     for image in jpgFilenamesList:
#         simImageCount += 1
#         #first write the file id which is same as the image id
#         sequence = os.path.basename(image).split('.')[0] + " " + "<start> " + songTempo + " " + timeSignature + listToString(notes) + " <end>" #append <start> and <end> tokens
#         #if current song is the last in the list, don't add a newline character to the end
#         if((len(songs)== songs.index(file) +1) and simImageCount == len(jpgFilenamesList)):
#             pass
#         else:
#             sequence = sequence + "\n"
#         midiData_file.write(sequence)

#   print(f"Finishes writing midi data. {songCount} songs written.")
#   #close file
#   midiData_file.close()


def load_file(fileName):
  """
  Args:
    fileName:

  Returns:
    content:
  """
  #open file in read mode
  file = open(fileName, 'r')
  #obtain content in file
  content = file.read()
  #close file
  file.close()

  return content


def dict_vocab_maxLength(content):
  #initialize dictionary
  midiDict = dict()
  vocabulary = set()
  maxLength = 0
  #loop over each line in content -> each midi song is separated by new line
  for line in content.split('\n'):
    #split line into tokens by white space
    tokens = line.split()
    #first token is image id, rest are midi events
    imageId, midi = tokens[0], tokens[1:]
    #create a list if not created
    if imageId not in midiDict:
      midiDict[imageId] = list()
    midiDict[imageId].append(midi)
    #build vocabulary
    vocabulary.update(midi)
    #store max length
    if len(tokens) > maxLength:
      maxLength = len(tokens)
  #return dict, vocabulary and maxlength
  return midiDict, vocabulary, maxLength


def tokenize(vocab):
  #filter 50 midi events with low frequency
  num_words= len(vocab) - 10
  #initialize tokenizer object without any filters
  tokenizer = Tokenizer(num_words=num_words, filters='')
  #generate tokens
  tokenizer.fit_on_texts(vocab)
  print("Number of tokens: " + str(len(tokenizer.word_index)))
  return tokenizer


content = load_file("midiData.txt")
train_midi, vocabulary, maxLength = dict_vocab_maxLength(content)
#print vocabulary length
vocabSize = len(vocabulary) + 1
print("Vocabulary length = " + str(vocabSize))
#print max length of sequence
print("Maximum length sequence = " + str(maxLength))
print(train_midi['0001'])
# print(vocabulary)
# maxLength
tokenizer = tokenize(vocabulary)
dump(tokenizer, open('tokenizer.pkl', 'wb'))
# tokenizer = load(open('tokenizer.pkl', 'rb'))


def extract_Image_feature(directory, modelName = current_ImgModel):
  #check for the model needed and initialize the model and image size
  if modelName == label_VGG16:
    model = VGG16()
    imgSize = (224, 224)
  else:
    model = InceptionV3(weights='imagenet')
    imgSize = (299, 299)

  #remove the last layer of the model to obtain the features
  #VGG16 has 4096 and InceptionV3 has 2048
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

  #print model summary
  print(model.summary())

  #intialize dictionary to extract features
  features = dict()

  #loop over each image in directory
  for imgName in listdir(directory):
    #get image path
    path = directory + '/' + imgName
    #load image and resize
    image = load_img(path, target_size=imgSize)
    #convert image to numpy array
    image = img_to_array(image)
    #reshape image to suit model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #preprocess image for the model
    image = preprocess_input(image)
    #extract features
    feature = model.predict(image, verbose=0)
    #get image id
    image_id = imgName.split('.')[0]
    #append to dictionary
    features[image_id] = feature

    #print image name
    print("Features extracted for " + imgName)

  return features


def load_featuresPickle(path):
  #load all features
  features = load(open(path, 'rb'))
  return features


def create_Model(vocabSize, maxLength):
  #image feature extractor model
  if current_ImgModel == label_VGG16:
    inputs1 = Input(shape = (4096, ))
  elif current_ImgModel == label_InceptionV3:
    inputs1 = Input(shape = (2048, ))

  fe1 = Dropout(0.5)(inputs1)
  fe2 = Dense(256, activation='relu')(fe1)

  #midi sequence model
  inputs2 = Input(shape=(maxLength,))
  #Embedding layer
  se1 = Embedding(vocabSize, 256, mask_zero=True)(inputs2)
  se2 = Dropout(0.2)(se1)
  se3 = LSTM(128, return_sequences=True)(se2)
  se3 = LSTM(256, return_sequences=True)(se3)
  se4 = Dropout(0.2)(se3)
  se5 = LSTM(256)(se4)

  #decoder model
  decoder1 = add([fe2, se5])
  decoder2 = Dense(128, activation='relu')(decoder1)
  decoder2 = Dense(256, activation='relu')(decoder2)
  outputs = Dense(vocabSize, activation='softmax')(decoder2)

  #Merger model
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  #compile model
  model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001))
  #print model summary
  print("Model Summary")
  model.summary()
  # plot_model(model, to_file='model.png', show_shapes=True)
  return model


def create_sequences(tokenizer, max_length, midi_list, photo, vocab_size):
  #initialize input lists
  X1, X2, y = list(), list(), list()
  #loop through each midi song for the image
  for midi in midi_list:
    #encode the midi sequence
    seq = tokenizer.texts_to_sequences([midi])[0]
    #generate multiple X,y pairs from one midi file
    for i in range(1, len(seq)):
      #generate input and output pair
      in_seq, out_seq = seq[:i], seq[i]
      #pad input sequence
      in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
      #encode output Sequence
      out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
      #append to input lists
      X1.append(photo)
      X2.append(in_seq)
      y.append(out_seq)

  return array(X1), array(X2), array(y)


def data_generator(midiData, photos, tokenizer, max_length, vocab_size):
  #loop forever
  while True:
    for key, midi_list in midiData.items():
      #get photo features
      photo = photos[key][0]
      in_img, in_seq, out_word = create_sequences(tokenizer, max_length, midi_list, photo, vocab_size)
      # yield [in_img, in_seq], out_word
      # print("in_img = ",in_img)
      # print("in_seq = ",in_seq)
      # print("out_word = ",out_word)
      yield (in_img, in_seq), out_word


def get_midiString_for_Integer(integer, tokenizer):
  #loop through tokenizer to find a match
  for midiString, index in tokenizer.word_index.items():
    if index == integer:
      return midiString
  return None


def generate_midiSequence(model, tokenizer, photo, maxLength):
  #create initial token
  midiSequence = '<start>' #string with midi events including <start> and <end> tokens
  prediction_list = [] #list with midi events without <start> and <end> tokens

  #iterate over max length of a sequence
  for i in range(maxLength):
    #encode sequence
    sequence =  tokenizer.texts_to_sequences([midiSequence])[0]
    #pad sequence
    sequence = pad_sequences([sequence], maxlen=maxLength)
    #predict next midi event string
    yhat = model.predict([photo,sequence], verbose=0)
    #print(yhat)
    #obtain event with highest probability
    #import numpy as np
    #yhat2 = np.argsort(np.max(yhat, axis=0))[-2]
    yhat = argmax(yhat)
    #map integer to midi string
    midiString = get_midiString_for_Integer(yhat, tokenizer)
    #stop if cannot find
    if midiString is None:
      break
    #append midiString to sequence
    if not midiString == '<start>':
        midiSequence += ' ' + midiString
    #stop if end of midi
    if midiString == '<end>':
      break
    #append midi event string to prediction list
    if not midiString == '<start>':
        prediction_list.append(midiString)

  return midiSequence, prediction_list


def create_midi(prediction_output, midiName):
  #initiate offset to 0
  offset = 0.0
  #initiate midi stram
  midi_stream = stream.Stream()

  #loop over each midiString patter in prediction Output
  for pattern in prediction_output:
    #Seperate midiString into event and time by @ symbol
    patternString = pattern.split('@')[0] #0 position stores midi event
    if pattern.split('@')[1] == 'tempo':
      #if event is a tempo, append to midiStream and continue to next iteration
      tp0 = tempo.MetronomeMark(patternString)
      tp0.setQuarterBPM(int(float(patternString)))
      midi_stream.append(tp0)
      continue
    #check if position 1 == time to check if event is a timeSignature
    if pattern.split('@')[1] == 'time':
      #if event is a timeSignature, append to midiStream and continue to next iteration
      ts0 = meter.TimeSignature(patternString)
      midi_stream.append(ts0)
      continue
    #check if '.' in patterString  or it patternString is a digit to detemine if the event is a chord
    if ('.' in patternString) or patternString.isdigit():
        #if event is a chord, obtain notes
        notes_in_chord = patternString.split('.')
        notes = []
        #loop for each note and create a notes list
        for current_note in notes_in_chord:
          new_note = note.Note(int(current_note))
          new_note.storedInstrument = instrument.Piano()
          notes.append(new_note)
        #create a chord using notes list
        new_chord = chord.Chord(notes)
        #set duration of chord
        new_chord.quarterLength = eval(pattern.split('@')[1])
        #update offset
        new_chord.offset = offset
        offset += new_chord.quarterLength
        #append chord to midi Stream
        midi_stream.append(new_chord)
    # pattern is a note or rest
    else:
      #if pattern is a rest
      if patternString == 'r':
        #create rest event
        new_note = note.Rest()
        #set duration
        new_note.quarterLength = eval(pattern.split('@')[1])
        #update offest
        new_note.offset = offset
        offset += new_note.quarterLength
        #append to midi stream
        midi_stream.append(new_note)
      else:
        #if pattern is a note
        #create note
        new_note = note.Note(patternString)
        #set note duration
        new_note.quarterLength = eval(pattern.split('@')[1])
        #update offset
        new_note.offset = offset
        offset += new_note.quarterLength
        new_note.storedInstrument = instrument.Piano()
        midi_stream.append(new_note)

  midi_stream.makeMeasures(inPlace = True)

  print('Saving Output file as midi....')

  midi_stream.write('midi', fp='testMidi_trans_incept/' + midiName + '.mid')


def extract_featuresPredict(path, modelName = current_ImgModel):
  #initialize model and resize image
  if modelName == label_VGG16:
    model = model = VGG16()
    image = load_img(path, target_size=(224, 224))
  elif modelName == label_InceptionV3:
    model = InceptionV3(weights='imagenet')
    # image = load_img(filename, target_size=(299, 299))

  #remove last layer of model
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
  #convert image to numpy array
  image = img_to_array(image)
  #reshape array
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  #preprocess image
  image = preprocess_input(image)
  #obtain features
  features = model.predict(image, verbose=0)

  return features

# features = extract_Image_feature('train_data/image')
# #print length of features
# print("Extracted features : %d" % len(features))
# # save to pickle file
# dump(features, open('features_incept.pkl', 'wb'))
train_features = load_featuresPickle('features_incept.pkl')
print(len(train_features))


#initialize epochs
epochs = 250
#initials steps to generate data
steps = len(train_midi)
#create model
# model = create_Model(vocabSize= vocabSize, maxLength=maxLength)
model = create_model_with_2_transformer(vocab_size= vocabSize, max_length=maxLength)
# model = create_model_with_4_transformer_MIDI(vocab_size= vocabSize, max_length=maxLength)


#save weights
# filepath = 'training/cp.ckpt'
# filepath = 'training_new_model/trans.weights.h5'
filepath = 'training_new_model/transincept.weights.h5'
checkpoint_dir = os.path.dirname(filepath)
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
for i in range(epochs):
  #create data generator
  generator  = data_generator(train_midi, train_features, tokenizer, maxLength, vocabSize)
  #fit model for one epoch
  print("Epoch No : " + str(i + 1))
  model.fit(generator, epochs=1, steps_per_epoch=steps, callbacks=[checkpoint], verbose=1)

# Epoch No : 1
# 2884/2884 [==============================] - ETA: 0s - loss: 6.8730
# Epoch 1: loss improved from inf to 6.87299, saving model to training_new_model/cp.weights.h5
# 2884/2884 [==============================] - 5680s 2s/step - loss: 6.8730
# Epoch No : 2
#  349/2884 [==>...........................] - ETA: 1:38:23 - loss: 6.7597

#load tokenizer from pickle
tokenizer = load(open('tokenizer.pkl', 'rb'))
#load best model
# model = create_Model(vocabSize, maxLength)
model = create_model_with_2_transformer(vocab_size= vocabSize, max_length=maxLength)
# filepath = 'training/cp.weights.h5'
model.load_weights(filepath)
# give the path to testImages, here testImages = image
images = glob('test/image/*.jpg')
print(images)
prv =0
ii = 0
prvtxt = ""
for img in images:
    photo = extract_featuresPredict(img)
    #generate midi
    midiSequence, prediction_output = generate_midiSequence(model, tokenizer, photo, maxLength)
    if(ii > 0) : 
        print(sum((photo - prv)[0]))
        prv = photo
    ii += 1
    print("Midi Sequnce in text")
    print(midiSequence)
    print(prvtxt == midiSequence)
    prvtxt = midiSequence

    #create midi file
    create_midi(prediction_output, img.split('.')[0].rsplit('/')[-1])

    