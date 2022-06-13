import numpy as np
from torch_scatter import scatter   
from config import *




# v13 91.00
#backbones = ['eca_nfnet_l2', 'tf_efficientnet_b7_ns']
#weights = [0.6, 0.4]


#backbones = ['eca_nfnet_l2', 'eca_nfnet_l1', 'tf_efficientnet_b7_ns']
#weights = [0.4, 0.3, 0.3]

backbones = ['eca_nfnet_l2', 'eca_nfnet_l1']
weights = [0.5, 0.5]



all_probs_ = [np.load(f'submits/features/{backbone}_probs.npy') for backbone in backbones]
all_observation_ids = np.load(f'submits/features/{backbones[0]}_observation_ids.npy')


N = all_probs_[0].shape[0]
num_classes = all_probs_[0].shape[1]

model_num = len(backbones)


all_probs = np.zeros(shape=[N, num_classes])

for i, probs in enumerate(all_probs_):
    all_probs += probs * weights[i]



### 合并同一个observation_id的
# sort
sort_index = np.argsort(all_observation_ids)

all_probs = all_probs[sort_index, :]

all_observation_ids = all_observation_ids[sort_index]




# scatter
unique_index = np.unique(all_observation_ids, return_index=True)[1]

inds = np.zeros(shape=N, dtype=np.int64)	

ind = 0
for i in range(len(unique_index)-1):
    start = unique_index[i]
    num = unique_index[i+1] - start
    inds[start:start+num] = ind
    ind += 1
inds[unique_index[-1]:] = ind	

unique_probs = scatter(torch.tensor(all_probs), torch.tensor(inds), dim=0, reduce='mean')
unique_observation_ids = all_observation_ids[unique_index]

unique_pred_probs, unique_pred_labels = torch.max(unique_probs, 1)
#print(unique_pred_probs.shape)
unique_pred_labels = unique_pred_labels.numpy()

N = unique_pred_labels.shape[0]

unique_observation_ids = unique_observation_ids.reshape(N, 1)
unique_pred_labels = unique_pred_labels.reshape(N, 1)
unique_pred_probs = unique_pred_probs.reshape(N, 1)
data_array = np.concatenate([unique_observation_ids, unique_pred_labels, unique_pred_probs], axis=1)



column_names_prob = ['ObservationId', 'class_id', 'prob']

submit_df = pd.DataFrame(data_array, columns=column_names_prob)
submit_df['ObservationId'] = submit_df['ObservationId'].astype(dtype='int')
submit_df['class_id'] = submit_df['class_id'].astype(dtype='int')


version = 'ensemble'


submit_df.to_csv(f'submits/submit_{version}_prob.csv', index=None)

del submit_df['prob']

submit_df.to_csv(f'submits/submit_{version}.csv', index=None)









