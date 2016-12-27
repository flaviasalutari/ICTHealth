import csv
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1234)#in order to get always the same results

df = pd.DataFrame(list(csv.reader(open("arrhythmia.csv","rb"), delimiter=','))).astype('float64')
df=df.loc[:, (df!=0).any(axis=0)]

classes_patients = df.iloc[:,-1].values

df_norm = (df - df.mean(axis = 0)) / df.std(axis = 0)
Nsamples, columuns= df_norm.shape
x=tf.placeholder(tf.float32,[Nsamples,1])#inputs
t=tf.placeholder(tf.float32,[Nsamples,1])#desired outputs
#--- neural netw structure:
w1=tf.Variable(tf.random_normal(shape=[1,3], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights"))
b1=tf.Variable(tf.random_normal(shape=[1,3], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases"))
a1=tf.matmul(x,w1)+b1
z1=tf.nn.sigmoid(a1)
w2=tf.Variable(tf.random_normal([3,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2=tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
y=tf.matmul(z1,w2)+b2# neural network output
#--- optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim=tf.train.GradientDescentOptimizer(2e-3,name="GradientDescent")# use gradient descent in the trainig phase
optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2])# minimize the objective function changing w1,b1,w2,b2
#--- initialize
init=tf.initialize_all_variables()
#--- run the learning machine
sess = tf.Session()
sess.run(init)
for i in range(10000):
    # generate the data
    xval=np.linspace(-1.0, 1.0, Nsamples)
    xval=np.reshape(xval, (Nsamples,1))
    tval=np.exp(-(np.square(xval)))
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))
print(sess.run(w2),sess.run(b2))