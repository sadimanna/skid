#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import shutil, time, os, random, copy, pickle, sys
from itertools import permutations 
from functools import reduce
import imageio
from skimage.transform import rotate, AffineTransform, warp, resize
import h5py

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, Reshape
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda, LeakyReLU, ConvLSTM2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers, activations
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as ppi_irv2

from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# ## LOADING THE PRED FILES

data_path = sys.argv[1]
model_path = sys.argv[2]
valid_mr_path = data_path + 'valid/'
#predictions_file = sys.argv[2]
print("Preparing the validation set labels...")
abn_df = pd.read_csv(data_path+'/valid-abnormal.csv',names=['patient_id','abn'])
acl_df = pd.read_csv(data_path+'/valid-acl.csv',names=['patient_id','acl'])
men_df = pd.read_csv(data_path+'/valid-meniscus.csv',names=['patient_id','men'])

dfs = [abn_df,acl_df,men_df]

valid_df = reduce(lambda left,right: pd.merge(left,right,on='patient_id'),dfs)

valid_df['filenames'] = valid_df.apply(lambda x : str(x.patient_id)+'.npy',axis=1)

print(valid_df.head(120))

axial_mode= 'axial'
sagit_mode='sagittal'
coron_mode='coronal'

NUM_FRAMES = 16
batch_size = 1
NUM_CLASSES = 3
NUM_PATCHES = 9

SEED = 16
SAMPLES = 8
NUM_SAMPLES = 120 

np.random.seed(16)

# ### PERFORMANCE METRICS

def TP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def TN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def FN(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def FP(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))

def get_accuracy(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    
    return (tp+tn)/(tp+fp+tn+fn)

def get_prevalence(y):
    return np.sum(y)/y.shape[0]

def sensitivity(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fn = FN(y,pred,th)
    
    return tp/(tp+fn)

def specificity(y, pred, th=0.5):
    tn = TN(y,pred,th)
    fp = FP(y,pred,th)
    
    return tn/(tn+fp)

def get_ppv(y, pred, th=0.5):
    tp = TP(y,pred,th)
    fp = FP(y,pred,th)
    
    return tp/(tp+fp)

def get_npv(y, pred, th=0.5):
    tn = TN(y,pred,th)
    fn = FN(y,pred,th)
    
    return tn/(tn+fn)


def get_performance_metrics(y, pred, class_labels, tp=TP,
                            tn=TN, fp=FP,
                            fn=FN,
                            acc=get_accuracy, prevalence=get_prevalence, 
                            spec=specificity,sens=sensitivity, ppv=get_ppv, 
                            npv=get_npv, auc=roc_auc_score, f1=f1_score,
                            thresholds=[]):
    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)

    columns = ["Condition", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity",
               "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        df.loc[i] = [class_labels[i],
        			 round(tp(y[:, i], pred[:, i]),3), 
        			 round(tn(y[:, i], pred[:, i]),3), 
        			 round(fp(y[:, i], pred[:, i]),3),
        			 round(fn(y[:, i], pred[:, i]),3),
        			 round(acc(y[:, i], pred[:, i], thresholds[i]),3),
        			 round(prevalence(y[:, i]),3),
        			 round(sens(y[:, i], pred[:, i], thresholds[i]),3),
        			 round(spec(y[:, i], pred[:, i], thresholds[i]),3),
        			 round(ppv(y[:, i], pred[:, i], thresholds[i]),3),
        			 round(npv(y[:, i], pred[:, i], thresholds[i]),3),
        			 round(auc(y[:, i], pred[:, i]),3),
        			 round(f1(y[:, i], pred[:, i] > thresholds[i]),3),
        			 round(thresholds[i], 3)]

    df = df.set_index("Condition")
    return df

def bootstrap_metric(y, pred, classes, metric='auc',bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))
    if metric=='AUC':
        metric_func = roc_auc_score
    if metric=='Sensitivity':
        metric_func = sensitivity
    if metric=='Specificity':
        metric_func = specificity
    if metric=='Accuracy':
        metric_func = get_accuracy
    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = metric_func(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

def get_confidence_intervals(y,pred,class_labels):
    
    metric_dfs = {}
    for metric in ['AUC','Sensitivity','Specificity','Accuracy']:
        statistics = bootstrap_metric(y,pred,class_labels,metric)
        df = pd.DataFrame(columns=["Mean "+metric+" (CI 5%-95%)"])
        for i in range(len(class_labels)):
            mean = statistics.mean(axis=1)[i]
            max_ = np.quantile(statistics, .95, axis=1)[i]
            min_ = np.quantile(statistics, .05, axis=1)[i]
            df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
        metric_dfs[metric] = df
    return metric_dfs

def conv_block(inp):
    out = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = None, 
                 kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(inp)
    #,kernel_regularizer =regularizers.l2(0.00001)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    out = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = None, 
                 kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    out = MaxPool2D(pool_size = (2,2),strides=2)(out)
    
    out = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = None,
                 kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    out = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = None,
                 kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    out = MaxPool2D(pool_size = (2,2),strides=2)(out)
    
    return out

def dim_red_block(inp,numfilts):
    out1 = Conv2D(filters = numfilts, kernel_size = 3, strides = 1, padding = 'same', activation = None ,
                  kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(inp)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)
    
    out1 = Conv2D(filters = numfilts, kernel_size = 3, strides = 2, padding = 'same', activation = None ,
                  kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(out1)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)
    
    out2 = AveragePooling2D(pool_size=(2,2),strides=2)(inp)
    
    out2 = Conv2D(filters = numfilts, kernel_size = 1, strides = 1, padding = 'same', activation = None ,
                  kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(out2)
    out2 = BatchNormalization()(out2)
    out2 = Activation('relu')(out2)
    
    out = Concatenate()([out1,out2])
    
    return out

def skip_block(inp,numfilts,scale = 1.0):
    out1 = Conv2D(filters = numfilts//2, kernel_size = 3, strides = 1, padding = 'same', activation = None ,
                  kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(inp)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)
    
    out1 = Conv2D(filters = numfilts, kernel_size = 3, strides = 1, padding = 'same', activation = None ,
                  kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(out1)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)
    
    final_out = scale*out1 + inp
    
    return final_out

def skip_block2(inp,numfilts,scale = 1.0):
    out1 = Conv2D(filters = numfilts, kernel_size = 3, strides = 1, padding = 'same', activation = None ,
                  kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(inp)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)
    
    out1 = Conv2D(filters = numfilts, kernel_size = 3, strides = 1, padding = 'same', activation = None ,
                  kernel_initializer = tf.keras.initializers.he_normal(seed=SEED))(out1)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)
    
    final_out = scale*out1 + inp
    
    return final_out

def rocket_model_branch(input_shape,num):
    branch_inp = Input(shape=input_shape)
    branch = conv_block(branch_inp)
    
    model = Model(inputs = branch_inp, outputs = branch, name = 'branch'+str(num))
    
    return model

def conv1x1_part(input_shape):

    conv1x1_inp = Input(shape=input_shape)
    
    model_stem = Conv2D(filters = 1024, kernel_size = 1, strides = 1, padding = 'same', activation = None, 
                        kernel_initializer = tf.keras.initializers.he_normal(seed=SEED),
                        kernel_regularizer =regularizers.l2(0.0001))(conv1x1_inp)
    model_stem = BatchNormalization()(model_stem)
    model_stem = Activation('relu')(model_stem)
    
    model = Model(inputs=conv1x1_inp,
                  outputs = model_stem, name = 'conv1x1_part')
    
    return model

def rocket_model_part2(input_shape):

    model_stem_inp = Input(shape=input_shape)
    
    model_stem = skip_block2(model_stem_inp,1024,scale=0.25)
    model_stem = dim_red_block(model_stem,1024)
    
    model = Model(inputs = model_stem_inp, outputs = model_stem, name = 'rocket_model_part2')
    
    return model

def rocket_model_part3(input_shape):

    model_stem_inp = Input(shape=input_shape)
    
    model_stem = skip_block2(model_stem_inp,2048,scale=0.25)
    model_stem = dim_red_block(model_stem,2048)
    
    model = Model(inputs = model_stem_inp, outputs = model_stem, name = 'rocket_model_part3')
    
    return model


# In[15]:

print("Building Pretext model...")
SEED = 16
branch1 = rocket_model_branch((256,256,3),1)
branch2 = rocket_model_branch((256,256,3),2)
branch3 = rocket_model_branch((256,256,3),3)
branch4 = rocket_model_branch((256,256,3),4)
branch5 = rocket_model_branch((256,256,3),5)
branch6 = rocket_model_branch((256,256,3),6)
branch7 = rocket_model_branch((256,256,3),7)
branch8 = rocket_model_branch((256,256,3),8)
branch9 = rocket_model_branch((256,256,3),9)
conv1x1 = conv1x1_part((64,64,2304))
pretext_part2 = rocket_model_part2((64,64,1024))
pretext_part3 = rocket_model_part3((32,32,2048))


# ### Manual Initialization
# ### Final Ensemble Stage

class DSDataGen(Sequence):
    def __init__(self, filenames_df, plane,preprocess_input = None, 
                 batch_size=1, data_aug = True, num_frames = NUM_FRAMES):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.filenames_df = filenames_df
        self.plane = plane
        self.preprocess_input = preprocess_input
        self.MAX_FRAMES = num_frames #NUM_FRAMES
        self.indices = list(range(self.filenames_df.shape[0]))
        
        self.data_aug = data_aug
        
        self.startind = 0
        self.endind = self.startind + self.batch_size
    
    def load_volume(self,file_idx):
        filepath = valid_mr_path+self.plane+'/'+self.filenames_df['filenames'].iloc[file_idx]
        #print(self.filenames_df['filenames'].iloc[file_idx])
        npy_file = np.load(filepath)
        return npy_file
    
    def get_frames(self,idx):
        image_volume = self.load_volume(idx)
        tot_frames = image_volume.shape[0]
        self.num_frames = np.min([tot_frames,self.MAX_FRAMES])
        ftnum = tot_frames//5
        fnum = self.num_frames//5
        
        frame_idxs = random.sample(list(range(ftnum)),fnum)+\
                     random.sample(list(range(ftnum,tot_frames-ftnum)),self.num_frames-2*fnum)+\
                     random.sample(list(range(tot_frames-ftnum,tot_frames)),fnum)
        #frame_idxs = random.sample(list(range(self.num_frames)),16)
        #ONLY FOR STANFORD MODEL
        frame_idxs = np.array(sorted(frame_idxs))

        frames = np.array([]).reshape((0,256,256,3))
        for n in range(self.num_frames):
            frame_idx = frame_idxs[n]
            frame = np.array(image_volume[frame_idx,:,:])
            frame = np.expand_dims(frame,axis=2)
            frame = np.append(frame,np.append(frame,frame,axis=2),axis=2)
            
            frames = np.append(frames,np.expand_dims(frame,axis=0),axis=0)
        #print(frames.shape)
        return frames
    
    def __len__(self):
        return len(self.filenames_df)*4
    
    def __getitem__(self,idx):
        
        self.startind = idx*self.batch_size
        self.endind = self.startind + self.batch_size
        idx = idx//SAMPLES
        #DECLARE VARIABLES
        batch_imgs = np.array([]).reshape((0,256,256,3))
        
        #ds_batch_labs = self.mllabs[['abn','acl','men']].iloc[idx].values.reshape((1,-1)).astype(np.float32)
        
        #GET CLIP FRAMES
        batch_imgs = np.append(batch_imgs,self.get_frames(idx),axis=0)

        sind = 0
        eind = 0
        tot_frames = batch_imgs.shape[0]

        #PREPROCESS FRAMES
        batch_imgs = self.preprocess_input(batch_imgs)
        #sagittal_rocket_imgs[i] = np.clip(sagittal_rocket_imgs[i] + np.random.normal(0,0.01,sagittal_rocket_imgs[i].shape),0,1.0)
        b1out = branch1(batch_imgs)
        b2out = branch2(batch_imgs)
        b3out = branch3(batch_imgs)
        b4out = branch4(batch_imgs)
        b5out = branch5(batch_imgs)
        b6out = branch6(batch_imgs)
        b7out = branch7(batch_imgs)
        b8out = branch8(batch_imgs)
        b9out = branch9(batch_imgs)
        
        bouts = np.concatenate([b1out,b2out,b3out,b4out,b5out,b6out,b7out,b8out,b9out],axis=3)
        
        b1out,b2out,b3out,b4out,b5out,b6out,b7out,b8out,b9out = [None]*9
        
        bouts = conv1x1(bouts)
        
        bouts = pretext_part2(bouts)
        
        bouts = pretext_part3(bouts)
        
        bouts = np.expand_dims(bouts,axis=0)
        
        return bouts

# SAGITTAL

def initialize(filename):

	branch1.load_weights(model_path + filename,by_name=True)
	branch2.load_weights(model_path + filename,by_name=True)
	branch3.load_weights(model_path + filename,by_name=True)
	branch4.load_weights(model_path + filename,by_name=True)
	branch5.load_weights(model_path + filename,by_name=True)
	branch6.load_weights(model_path + filename,by_name=True)
	branch7.load_weights(model_path + filename,by_name=True)
	branch8.load_weights(model_path + filename,by_name=True)
	branch9.load_weights(model_path + filename,by_name=True)
	conv1x1.load_weights(model_path + filename,by_name=True)
	pretext_part2.load_weights(model_path + filename,by_name=True)
	pretext_part3.load_weights(model_path + filename,by_name=True)


def predictions(valid_df,vdg):
	pred = np.array([]).reshape((0,3))
	jpp_pred = np.array([]).reshape((0,3))
	t = 0
	for i in range(SAMPLES*NUM_SAMPLES):
	    i1 = vdg.__getitem__(i)
	    jpp_temp_pred = dsmodel(i1)
	    jpp_pred = np.append(jpp_pred,jpp_temp_pred.numpy(),axis=0)
	    t+=1
	    if t==8:
        	t = 0
	        pred = np.append(pred,np.mean(jpp_pred,axis=0,keepdims=True),axis=0)
	        jpp_pred = np.array([]).reshape((0,3))
	
	return pred

# PREDICTIONS AND ENSEMBLING
class_labels = ['abnormality','acl tear','meniscus tear']

#--------------------SAGITTAL-------------
print("Initializing Sagittal Pretext Model...")
initialize("pretext/sagittal_pretext_model.h5")
vdg = DSDataGen(valid_df, preprocess_input = ppi_irv2, plane = 'sagittal',batch_size=1, data_aug = False, num_frames = NUM_FRAMES)
dsmodel = tf.keras.models.load_model(model_path+'downstream/sagittal_downstream_model.h5',compile=False)

print("Generating Predictions on Sagittal Plane of Validation Set...")
sag_pred = predictions(valid_df,vdg)

sag_perf_df = get_performance_metrics(valid_df[['abn','acl','men']].values,sag_pred,class_labels)

#--------------------CORONAL-----------------
print("Initializing Coronal Pretext Model...")
initialize("pretext/coronal_pretext_model.h5")
vdg = DSDataGen(valid_df, preprocess_input = ppi_irv2, plane = 'coronal',batch_size=1, data_aug = False, num_frames = NUM_FRAMES)
dsmodel = tf.keras.models.load_model(model_path+'downstream/coronal_downstream_model.h5',compile=False)
print("Generating Predictions on Coronal plane of Validation Set...")
cor_pred = predictions(valid_df,vdg)

cor_perf_df = get_performance_metrics(valid_df[['abn','acl','men']].values,cor_pred,class_labels)

#--------------------AXIAL-----------------
print("Initializing Axial Pretext Model...")
initialize("pretext/axial_pretext_model.h5")
vdg = DSDataGen(valid_df, preprocess_input = ppi_irv2, plane = 'axial',batch_size=1, data_aug = False, num_frames = NUM_FRAMES)
dsmodel = tf.keras.models.load_model(model_path+'downstream/axial_downstream_model.h5',compile=False)
print("Generating Predictions on Axial plane of Validation Set...")
axi_pred = predictions(valid_df,vdg)

axi_perf_df = get_perfromance_metrics(valid_df[['abn','acl','men']].values,axi-pred,class_labels)

#----------------ENSEMBLE--------------

#Validation accuracies obtained on the MRNet dataset

'''# ACCURACIES GIVEN IN THE PAPER
sag_abn_acc = 0.883
sag_acl_acc = 0.750
sag_men_acc = 0.642

cor_abn_acc = 0.867
cor_acl_acc = 0.708
cor_men_acc = 0.692

axi_abn_acc = 0.850
axi_acl_acc = 0.825
axi_men_acc = 0.650'''

sag_abn_acc, sag_acl_acc, sag_men_acc = sag_perf_df[['Accuracy']].values
cor_abn_acc, cor_acl_acc, cor_men_acc = cor_perf_df[['Accuracy']].values
axi_abn_acc, axi_acl_acc, axi_men_acc = axi_perf_df[['Accuracy']].values

print("Generating weights for Ensemble model...")

w_abn = [np.log(sag_abn_acc/(1-sag_abn_acc)),
         np.log(cor_abn_acc/(1-cor_abn_acc)),
         np.log(axi_abn_acc/(1-axi_abn_acc))]

w_acl = [np.log(sag_acl_acc/(1-sag_acl_acc)),
         np.log(cor_acl_acc/(1-cor_acl_acc)),
         np.log(axi_acl_acc/(1-axi_acl_acc))]

w_men = [np.log(sag_men_acc/(1-sag_men_acc)),
         np.log(cor_men_acc/(1-cor_men_acc)),
         np.log(axi_men_acc/(1-axi_men_acc))]
         
w_abn = w_abn / np.sum(w_abn)
w_acl = w_acl / np.sum(w_acl)
w_men = w_men / np.sum(w_men)

class_labels = ['abnormality','acl tear','meniscus tear']

abn_pred = []
acl_pred = []
men_pred = []

for i in range(NUM_SAMPLES):
    abn_pred.append(sag_pred[i,0]*w_abn[0]+cor_pred[i,0]*w_abn[1]+axi_pred[i,0]*w_abn[2])
    acl_pred.append(sag_pred[i,1]*w_acl[0]+cor_pred[i,1]*w_acl[1]+axi_pred[i,1]*w_acl[2])
    men_pred.append(sag_pred[i,2]*w_men[0]+cor_pred[i,2]*w_men[1]+axi_pred[i,2]*w_men[2])
    
abn_pred = np.array(abn_pred).reshape(NUM_SAMPLES,1)
acl_pred = np.array(acl_pred).reshape(NUM_SAMPLES,1)
men_pred = np.array(men_pred).reshape(NUM_SAMPLES,1)
preds = np.append(np.append(abn_pred,acl_pred,axis=1),men_pred,axis=1)

#preds_df = pd.DataFrame(data = preds,columns=None,index=None)

#preds_df.to_csv('predictions.csv',columns=None,header=None,index=False,index_label=None)

#METRICS
print("\nPerformance Metrics on Sagittal plane :\n")
print(sag_perf_df)

print("\nPerformance Metrics on Coronal plane :\n")
print(cor_perf_df)

print("\nPerformance Metrics on Axial plane :\n")
print(axi_perf_df)

print("\nEnsemble Results...\n")

perf_metrics = get_performance_metrics(valid_df[['abn','acl','men']].values,preds,class_labels)

print(perf_metrics)

conf_int_df = get_confidence_intervals(valid_df[['abn','acl','men']].values,preds,class_labels)

print("\nAccuracy :\n")
print(conf_int_df['Accuracy'])

print("\nSensitivity :\n")
print(conf_int_df['Sensitivity'])

print("\nSpecificity :\n")
print(conf_int_df['Specificity'])

print("\nAUC :\n")
print(conf_int_df['AUC'])
