import numpy as np
import pickle
from avg_corr.main import train as train_mse, PPOBuffer

random_weight = 2.0
length =100
seed = 9
env='Hopper-v4'
name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', length,
        'buffer_size', 16000, 'seed', seed , 'env', env]
name = '-'.join(str(x) for x in name)

with open('./dataset/' + name + '.pkl', 'rb') as outp:
    buf = pickle.load(outp)

gamma = 0.95
data = buf.sample(512,1)
tim, prod = data['tim'], data['prod']
logtarg, logbev = data['logtarg'], data['logbev']

label = np.exp(np.log(gamma) * tim + prod)

print(label[np.random.randint(low=0,high=512,size=10)])