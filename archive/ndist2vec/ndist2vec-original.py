import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import datetime
starttime = datetime.datetime.now()
### Data name and embedding dimension and number of landmark node ###
graph_name='Surat'
emb_size=50
num_landmark=512

### Read the shortest distance matrix file(.npy) ###
sdm = np.load("./data/%s_shortest_distance_matrix.npy"%graph_name)
maxLengthy = np.max(sdm)
n= sdm.shape[0]
nodes_index=np.random.permutation(n)

### Storage address ###
folder2 = './result./%s_res'% graph_name
folder1 ='./model./%s_models'% graph_name
if not os.path.exists(folder1):
    os.makedirs(folder1)
if not os.path.exists(folder2):
    os.makedirs(folder2)

### function ###
def get_sample_batch(landmark_nodes,remain_nodes):
    l1 = len(landmark_nodes)
    l2 = len(remain_nodes)
    x1_batch = np.zeros((l1*l2,))
    x2_batch = np.zeros((l1*l2,))
    y_batch = np.zeros((l1*l2, 1))
    z = 0
    for i in landmark_nodes:
        for j in remain_nodes:
            x1_batch[z]=i
            x2_batch[z]=j
            y_batch[z] = sdm[i][j]
            z += 1
    return x1_batch, x2_batch, y_batch

def get_all_batch(index_list):
    l = len(index_list)
    x1_batch = np.zeros((l,))
    x2_batch = np.zeros((l,))
    y_batch = np.zeros((l, 1))
    z = 0
    for i in index_list:
        node1 = int(i // (n - 1))
        node2 = i % (n - 1)
        if node2 >= node1:
            node2 += 1
        x1_batch[z] = node1
        x2_batch[z] = node2
        y_batch[z] = sdm[node1][node2]
        z += 1
    return x1_batch, x2_batch, y_batch

def get_eval_batch(node_index,num):
    lnode =len(node_index)
    x1_batch = np.zeros((int(lnode-num-1),))
    x2_batch = np.zeros((int(lnode-num-1),))
    y_batch = np.zeros((int(lnode-num-1), 1))
    z = 0
    for i in range(num+1,lnode):
        x1_batch[z] = num
        x2_batch[z]= i
        y_batch[z] = sdm[num][i]
        z=z+1
    return x1_batch, x2_batch, y_batch

### Build neural network ###
# Parameters #
learning_rate = 0.01
training_epochs = 30
n_hidden_1 = int(emb_size)
n_hidden_2 = 100
n_hidden_3 = 20
n_input = n
n_output=1

# Define neural network structure #
x1 = tf.placeholder("int32", [None, ], name="x1")
x2 = tf.placeholder("int32", [None, ], name="x2")
y = tf.placeholder("float32", [None, 1], name="y")
# lr = tf.placeholder("float32")
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], mean=0.0, stddev=0.01, dtype=tf.float32), name='h1'),

    'h21': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h21'),
    'h31': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h31'),
    'out1': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout1'),

    'h22': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h22'),
    'h32': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h32'),
    'out2': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout2'),

    'h23': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h23'),
    'h33': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h33'),
    'out3': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout3'),

    'h24': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h24'),
    'h34': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32),
                       name='h34'),
    'out4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], mean=0.0, stddev=0.01, dtype=tf.float32),
                        name='wout4'),
    'v1': tf.Variable(tf.random_uniform(shape=(1,), minval=0, maxval=100, dtype=tf.float32), name='v1'),
    'v2': tf.Variable(tf.random_uniform(shape=(1,), minval=100, maxval=1000, dtype=tf.float32), name='v2'),
    'v3': tf.Variable(tf.random_uniform(shape=(1,), minval=1000, maxval=10000, dtype=tf.float32), name='v3'),
    'v4': tf.Variable(tf.random_uniform(shape=(1,), minval=10000, maxval=(maxLengthy - 10000.).astype(np.float32),
                                        dtype=tf.float32),name='v4'),
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1], mean=0.0, stddev=0.01, dtype=tf.float32), name='b1'),

    'b21': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b21'),
    'b31': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b31'),
    'out1': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout1'),

    'b22': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b22'),
    'b32': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b32'),
    'out2': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout2'),

    'b23': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b23'),
    'b33': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b33'),
    'out3': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout3'),

    'b24': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.0, stddev=0.01, dtype=tf.float32), name='b24'),
    'b34': tf.Variable(tf.truncated_normal([n_hidden_3], mean=0.0, stddev=0.01, dtype=tf.float32), name='b34'),
    'out4': tf.Variable(tf.truncated_normal([n_output], mean=0.0, stddev=0.01, dtype=tf.float32), name='bout4'),
}
# function #
def multilayer_perceptron(x1, x2, weights, biases):
    # embedding layer
    layer_11 = tf.add(tf.gather(weights['h1'], x1),biases['b1'])
    layer_12 = tf.add(tf.gather(weights['h1'], x2),biases['b1'])
    layer_1 = tf.concat([layer_11, layer_12], 1)
    # first (0,a1)
    layer_21 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h21']), biases['b21']))
    layer_31 = tf.nn.relu(tf.add(tf.matmul(layer_21, weights['h31']), biases['b31']))
    out_layer1 = tf.sigmoid(tf.add(tf.matmul(layer_31, weights['out1']), biases['out1']))
    # second (0,a2)
    layer_22 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h22']), biases['b22']))
    layer_32 = tf.nn.relu(tf.add(tf.matmul(layer_22, weights['h32']), biases['b32']))
    out_layer2 = tf.sigmoid(tf.add(tf.matmul(layer_32, weights['out2']), biases['out2']))
    # third (0,a3)
    layer_23 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h23']), biases['b23']))
    layer_33 = tf.nn.relu(tf.add(tf.matmul(layer_23, weights['h33']), biases['b33']))
    out_layer3 = tf.sigmoid(tf.add(tf.matmul(layer_33, weights['out3']), biases['out3']))
    # forth (0,a4 )
    layer_24 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h24']), biases['b24']))
    layer_34 = tf.nn.relu(tf.add(tf.matmul(layer_24, weights['h34']), biases['b34']))
    out_layer4 = tf.sigmoid(tf.add(tf.matmul(layer_34, weights['out4']), biases['out4']))

    a11 = tf.multiply(out_layer1, weights['v1'])
    a12 = tf.multiply(out_layer2, weights['v2'])
    a21 = tf.multiply(out_layer3, weights['v3'])
    a22 = tf.multiply(out_layer4, weights['v4'])
    a1 = tf.add(a11, a12)
    a2 = tf.add(a21, a22)
    out_layer = tf.add(a1, a2)
    return out_layer

# Construct model #
pred = multilayer_perceptron(x1, x2, weights, biases)

# Define loss and optimizer
cost = tf.losses.mean_squared_error(y, pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.add_to_collection("optimizer", optimizer)

# Initializing the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
print ("neural network has been built")

### Training ###
for epoch in range(training_epochs):
    avg_cost = 0.
    if epoch==0:
        total_batch = n
        random_index = np.random.permutation(n*(n-1))
        for j in range(total_batch):
            start = j * n
            end = (j + 1) * n
            if end >= n*(n-1):
                end = n*(n-1)
            batch_x1, batch_x2, batch_y = get_all_batch(random_index[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x1: batch_x1,
                                                      x2: batch_x2,
                                                      y: batch_y,
                                                      })
            avg_cost += c / total_batch

    else:
        random_index = np.random.permutation(n)
        landmark_nodes = random.sample(list(random_index), num_landmark)
        remain_nodes = list(set(list(random_index)) - set(landmark_nodes))
        lens_l = len(landmark_nodes)
        lens_r = len(remain_nodes)
        size = 100*int(n // lens_l) + 1
        total_batch = int(lens_r // size) + 1
        random_nodes = random.sample(remain_nodes, lens_r)
        for j in range(total_batch):
            start = j * size
            end = (j + 1) * size
            if end >= lens_r:
                end = lens_r
            batch_x1, batch_x2, batch_y = get_sample_batch(landmark_nodes, random_nodes[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x1: batch_x1,
                                                          x2: batch_x2,
                                                          y: batch_y})
            avg_cost += c / total_batch
    save_path = saver.save(sess, folder1 + '/model.ckpt')
    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
print("Optimization Finished!")
endtime = datetime.datetime.now()
Running_time=endtime - starttime
### Predict all node pairs ###
starttime = datetime.datetime.now()
result = []
real_dis = []
for i in range(n):
    batch_x1, batch_x2, batch_y = get_eval_batch(nodes_index, i)
    result_temp = sess.run(pred, feed_dict={x1: batch_x1, x2:batch_x2})
    result = np.append(result, result_temp)
    real_dis = np.append(real_dis, batch_y)
endtime = datetime.datetime.now()

arr1=real_dis.astype(int)
arr2=result.astype(int)
ac=np.equal(arr1, arr2).sum()/len(arr1)

abe = np.fabs(real_dis - result)
re = abe/real_dis

mse = (abe ** 2).mean()
maxe = np.max(abe ** 2)
mine = np.min(abe ** 2)
mabe = abe.mean()
maxae = np.max(abe)
minae = np.min(abe)
mre = re.mean()
maxre = np.max(re)
minre = np.min(re)
Predict_time=endtime - starttime
print ("mean square error:", mse)
print ("max square error:", maxe)
print ("min square error:", mine)
print ("mean absolute error:", mabe)
print ("max absolute error:", maxae)
print ("min absolute error:", minae)
print ("mean relative error:", mre)
print ("max relative error:", maxre)
print ("min relative error:", minre)
print('Running time:',Running_time)
print('Predict time:',Predict_time)

### Storage result ###
f = open(folder2+'/%s_res.txt' %graph_name, 'w')
f.write('accuracy:'+''+ str(ac) + '\n'
        +"mean square error:"+''+ str(mse) + '\n'
        +"max square error:"+''+ str(maxe) + '\n'
        +"min square error:"+''+ str(mine) + '\n'
        +"mean absolute error:"+''+ str(mabe) + '\n'
        +"max absolute error:"+''+ str(maxae) + '\n'
        +"min absolute error:"+''+ str(minae) + '\n'
        +"mean relative error:"+''+ str(mre) + '\n'
        +"max relative error:"+''+ str(maxre) + '\n'
        +"min relative error:"+''+ str(minre) + '\n'
        +"Running time:"+''+ str(Running_time) + '\n'
        +"Predict time:"+''+str(Predict_time)+'\n')