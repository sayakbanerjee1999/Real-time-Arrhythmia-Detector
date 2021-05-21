# Importing the Libraries for pre-processing
import os
import scipy.io
import scipy 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#df = pd.read_csv("REFERENCE-original.csv", header = None)
#df.to_csv("REFERENCE-original.csv", header = ["ECG ID", "ECG LABEL"], index = False)

df = pd.read_csv("REFERENCE-original.csv")
df.head()
df['ECG LABEL'].value_counts()



dir = "training2017/"
samples = []
N = 7500                #Trim upto N Samples - Change this to 7500

for filename in os.listdir(dir):
    if filename.endswith('.mat'):
        mat_data = scipy.io.loadmat('training2017//' + filename)
        record = mat_data['val'].tolist()[0]
        record = (record + N * [0])[ : N]               #Padding and Trimming
        samples.append(record)
        
dataframe = pd.DataFrame(samples)
arr = np.array(samples)

# X = dataframe

y = df['ECG LABEL'].tolist()

for i in range(len(y)):
    if y[i] == "N":
        y[i] = 0
    elif y[i] == "O":
        y[i] = 1
    elif y[i] == "A":
        y[i] = 2
    elif y[i] == "~":
        y[i] = 3
        
Y = np.array(y)
#Y = pd.DataFrame(y)




'''Normalising the Sampled Values'''
import statistics

# Subtract Mean
for i in range(dataframe.shape[0]):
    avg = statistics.mean(samples[i])
    for j in range(N):
        samples[i][j] = samples[i][j] - avg
    
dataframe = pd.DataFrame(samples)
#arr = np.array(samples)


#Divide by the Mod of the Abs Value
for i in range(dataframe.shape[0]):
    res = [abs(ele) for ele in samples[i]]
    gre = max(res)
    for j in range(N):
        samples[i][j] = samples[i][j]/gre
        
dataframe = pd.DataFrame(samples)
X = np.array(samples)

# Normal - N,    Others - O,    Atrial Fibrillation - A,     ~ - Noisy 
dict = {"N": 0, "O": 1, "A": 2, "~": 3}




'''Pass Sampled Value through Filter after Checking Accuracy'''






'''Test Train Split'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)




'''SMOTE Oversampling for Training Data'''
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
X_train, Y_train = sm.fit_resample(X_train, Y_train)





'''Applying SVM ML Model'''
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(X_train, Y_train)


'''ML Predictions'''
ml_Y_pred = classifier.predict(X_test)
#ml_Y_pred = np.argmax(pred, axis = 1)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(Y_test, ml_Y_pred)
cm = pd.DataFrame(cm)
cm.columns = ['Normal', 'Others', 'Atrial Fibrillation', 'Noisy']
cm.index = ['Normal', 'Others', 'Atrial Fibrillation', 'Noisy']


acc = accuracy_score(Y_test, ml_Y_pred)
report = classification_report(Y_test, ml_Y_pred) 

print(cm)
print(acc*100,"%")
print(report)





'''Encoding Categorical Output Data'''
from keras.utils import to_categorical
Y_train_cat = to_categorical(Y_train, num_classes = 4)
Y_test_cat = to_categorical(Y_test, num_classes = 4)




'''Expanding the Dimensions to match Input Shape''' 
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)





'''Model Goes Here'''
import tensorflow as tf
from keras.models import Model 
from keras.layers import Input, LSTM, MaxPooling1D, Conv1D, Dense, Dropout, BatchNormalization
from keras.layers import ReLU
from keras.callbacks import History


history = History()


input = Input(shape = X_train[0].shape)

x = Conv1D(64, kernel_size = 10)(input)
x = ReLU()(x)
x = MaxPooling1D(2)(x)


x = Conv1D(32, kernel_size = 6)(input)
x = ReLU()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.05)(x)

x = Conv1D(32, kernel_size = 5)(x)
x = ReLU()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.10)(x)

x = Conv1D(16, kernel_size = 3)(x)
x = ReLU()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.15)(x)


x = LSTM(units = 64, activation = "tanh", return_sequences = True)(x)
x = LSTM(units = 64, activation = "tanh")(x)

x = Dense(units = 4, activation = "softmax")(x)


model = Model(inputs = input, outputs = x)

'''Compile and Fit the Model'''
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
        
batch_size = 32
steps_per_epoch = len(X_train)//batch_size

m = model.fit(X_train, Y_train_cat, validation_data = (X_test, Y_test_cat),
                   epochs = 100, batch_size = 32, callbacks = [history])
        


#Save the Model that Gives High Acccuracy and Good F1 Score
#model_name = 'ecg.h5'
#model.save(model_name)




'''Load the Trained - Model'''
from keras.models import load_model
model = load_model('ecg_75.h5')

model.summary()



'''Accuracy, Loss Plot'''
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')





'''Predictions'''
pred = model.predict(X_test)
Y_pred = np.argmax(pred, axis = 1)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(Y_test, Y_pred)
cm = pd.DataFrame(cm)
cm.columns = ['Normal', 'Others', 'Atrial Fibrillation', 'Noisy']
cm.index = ['Normal', 'Others', 'Atrial Fibrillation', 'Noisy']


acc = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred) 

print(cm)
print(acc*100,"%")
print(report)








'''Sending Predicted Output to Thingspeak/Firebase'''
#Predicting Single Output
import random

check = random.randint(0, 852)

input_data = X_test[check]
input_data = np.expand_dims(input_data, axis = 0)

predict_output = model.predict(input_data)

output = np.argmax(predict_output, axis = 1)[0]
output = 0



'''Mapping Output with Result & Notification''' 
result = ""

if output == 0:
    result = "Normal Sinus Rhythm"
elif output == 1:
    result = "Others"
elif output == 2:
    result = "Atrial Fibrillation"
elif output == 3:
    result = "Noisy ECG Sample"
    


notification = ""

if output == 0:
    notification = "You are having Normal Sinus Rhythm. Nothing to worry about. It's all fine."
elif output == 1:
    notification = "You are having a different type of Sinus Rhythm. But there is nothing to worry about"
elif output == 2:
    notification = "You ECG Sample is confirmed with Atrial Fibrillation. Please fix an appoinement with a doctor and take prescribed medicines as early as possible."
elif output == 3:
    notification = "The ECG sample is Noisy. Please undergo the ECG sampling procedure again to get a better sample signal to perform the test on."





# Sending Data to Thingspeak
from urllib import request as req

url = 'https://api.thingspeak.com/update?api_key=AM5JLTTWSAF7UWY8&field1='   +   str(output)
write = req.urlopen(url)
print(write.read())