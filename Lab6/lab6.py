import csv
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1234)#in order to get always the same results

df = pd.DataFrame(list(csv.reader(open("parkinsoncsv.csv","rb"), delimiter=','))).astype('float64')

df_norm = (df - df.mean(axis = 0)) / df.std(axis = 0)
col_estimate = 5
y_train = df_norm[col_estimate].values.reshape(len(df_norm[col_estimate]),1)
df_norm = df_norm.drop(df_norm.columns[[0, 3, 4, 5, 6]], axis=1)
rig,col=df_norm.shape
Nsamples = rig
x=tf.placeholder(tf.float32,[Nsamples,col])#inputs
t=tf.placeholder(tf.float32,[Nsamples,1])#desired outputs
#--- neural netw structure:
w1=tf.Variable(tf.random_normal(shape=[col,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights"))
b1=tf.Variable(tf.random_normal(shape=[Nsamples,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases"))
a1=tf.matmul(x,w1)+b1
y = a1
cost=tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim=tf.train.GradientDescentOptimizer(2e-9,name="GradientDescent")# use gradient descent in the trainig phase
optim_op = optim.minimize(cost, var_list=[w1,b1])# minimize the objective function changing w1,b1,w2,b2
#--- initialize
init=tf.initialize_all_variables()
#--- run the learning machine
sess = tf.Session()
sess.run(init)

# generate the data
xval=df_norm.values
tval=y_train
for i in range(10000):
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))

yval=y.eval(feed_dict=train_data,session=sess)
plt.plot(tval,'ro', label='regressand')
plt.plot(yval,'bx', label='regression')
plt.xlabel('case number')
plt.grid(which='major', axis='both')
plt.legend()
plt.savefig('one.pdf',format='pdf')
plt.show()











