import os 
import numpy as np

def save_results(y_pred,y_true, samples,trace,CROSS_VAL_DIR_NAME):
    BASE_PATH = os.getcwd()
    #NEW_PATH = 'TODO: descriptive name for current run of cross validation'
    NEW_PATH = os.path.join(BASE_PATH,CROSS_VAL_DIR_NAME) 
    if not os.path.isdir(NEW_PATH): os.mkdir(NEW_PATH)
    save_samples(samples,NEW_PATH)
    save_predictions(y_pred,y_true,NEW_PATH)
    save_PE(trace,NEW_PATH)
    
def save_samples(samples,NEW_PATH):
    #samples is tuple of length 2*num_layers. (i) is 2d array of weights (i+1) is 2d array of biases
    #samples[i] is tensorflow.python.framework.ops.EagerTensor of size (num_samples,prev_layer, curr_layer), num_samples is number of hmc samples
    for i in range(0, len(samples)-1, 2):
        curr_layer_num = str(int(i/2))
        weights_file_path = os.path.join(NEW_PATH,'weights_layer_'+curr_layer_num)
        biases_file_path = os.path.join(NEW_PATH,'biases_layer_'+curr_layer_num)
        if not os.path.isdir(weights_file_path): os.mkdir(weights_file_path)
        if not os.path.isdir(biases_file_path) : os.mkdir(biases_file_path)
        num_hmc_samples = samples[i].shape[0]
        for curr_sample_num in range(num_hmc_samples):
            curr_weights = samples[i][curr_sample_num]
            curr_biases = samples[i+1][curr_sample_num]
            curr_weights_fname = os.path.join(weights_file_path,'weights_layer_'+str(curr_layer_num)+'_sample_'+str(curr_sample_num)+'.csv')
            curr_biases_fname = os.path.join(biases_file_path,'biases_layer_'+str(curr_layer_num)+'_sample_'+str(curr_sample_num)+'.csv')
            #np.savetxt(curr_weights_fname, curr_weights, delimiter=",")
            #np.savetxt(curr_biases_fname, curr_biases, delimiter=",")
            
def save_predictions(y_pred,y_true,NEW_PATH):
    #Note: y_pred is list of arrays. y_true is array 
    PRED_PATH = os.path.join(NEW_PATH,'predictions')
    if not os.path.isdir(PRED_PATH): os.mkdir(PRED_PATH)
    for i,prediction_arr in enumerate(y_pred):
        pred_fname = os.path.join(PRED_PATH,'pred_'+str(i)+'.csv')
        np.savetxt(pred_fname, prediction_arr, delimiter=",")
    true_fname = os.path.join(PRED_PATH,'y_true'+'.csv')
    np.savetxt(true_fname, y_true, delimiter=",")

def save_PE(trace,NEW_PATH):
    fname = os.path.join(NEW_PATH,'negLogProb.csv')
    t_log_prob = trace[0].inner_results.accepted_results.target_log_prob
    np.savetxt(fname, t_log_prob, delimiter=",")
    
