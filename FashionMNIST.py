import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def conv2d(x,W,b,strides=1):
  
  x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
  x = tf.nn.bias_add(x,b)
  
  return tf.nn.relu(x)

def maxpool2d(x,k=2):
  
  return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def conv_net(x,weights,baises):
  
  conv1 = conv2d(x,weights['wc1'],baises['b1'])
  conv1 = maxpool2d(conv1)
  
  conv2 = conv2d(conv1,weights['wc2'],baises['b2'])
  conv2 = maxpool2d(conv2)
  
  conv2 = tf.nn.dropout(conv2,0.2)
  
  conv3 = conv2d(conv2,weights['wc3'],baises['b3'])
  conv3 = maxpool2d(conv3)
  
  fc1 = tf.reshape(conv3,[-1,weights['wd1'].get_shape().as_list()[0]])
  fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
  fc1 = tf.nn.relu(fc1)
  fc1 = tf.nn.dropout(fc1,0.5)
  
  return tf.add(tf.matmul(fc1,weights['out']),biases['out'])


data = input_data.read_data_sets('data/fashion', one_hot="True")

X_train = data.train.images
y_train = data.train.labels
X_test = data.test.images
y_test = data.test.labels

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

n_classes = 10
n_iterations = 150
learning_rate = 0.001
batch_size = 128
n_input = 28

x = tf.placeholder("float",[None,28,28,1])
y = tf.placeholder("float",[None,n_classes])


'''
weights = {
          'wc1':tf.get_variable('W7',shape=(3,3,1,32),initializer=tf.contrib.layers.xavier_initializer()),
          'wc2':tf.get_variable('W8',shape=(3,3,32,64),initializer=tf.contrib.layers.xavier_initializer()),
          'wc3':tf.get_variable('W9',shape=(3,3,64,128),initializer=tf.contrib.layers.xavier_initializer()),
          'wd1':tf.get_variable('W10',shape=(4*4*128,128),initializer=tf.contrib.layers.xavier_initializer()),
          'out':tf.get_variable('W6',shape=(128,n_classes),initializer=tf.contrib.layers.xavier_initializer())
          }

biases = {
          'b1':tf.get_variable('B0',shape=(32),initializer=tf.contrib.layers.xavier_initializer()),
          'b2':tf.get_variable('B1',shape=(64),initializer=tf.contrib.layers.xavier_initializer()),
          'b3':tf.get_variable('B2',shape=(128),initializer=tf.contrib.layers.xavier_initializer()),
          'bd1':tf.get_variable('B3',shape=(128),initializer=tf.contrib.layers.xavier_initializer()),
          'out':tf.get_variable('B4',shape=(10),initializer=tf.contrib.layers.xavier_initializer())
         }
'''



weights = {
          'wc1':tf.Variable(tf.truncated_normal(shape=(3,3,1,32),mean=0, stddev=0.08)),
          'wc2':tf.Variable(tf.truncated_normal(shape=(3,3,32,64),mean=0, stddev=0.08)),
          'wc3':tf.Variable(tf.truncated_normal(shape=(3,3,64,128),mean=0, stddev=0.08)),
          'wd1':tf.Variable(tf.truncated_normal(shape=(4*4*128,128),mean=0, stddev=0.08)),
          'out':tf.Variable(tf.truncated_normal(shape=(128,n_classes),mean=0, stddev=0.08))
          }

biases = {
          'b1':tf.Variable(tf.truncated_normal(shape=[32],mean=0,stddev=0.08)),
          'b2':tf.Variable(tf.truncated_normal(shape=[64],mean=0,stddev=0.08)),
          'b3':tf.Variable(tf.truncated_normal(shape=[128],mean=0,stddev=0.08)),
          'bd1':tf.Variable(tf.truncated_normal(shape=[128],mean=0,stddev=0.08)),
          'out':tf.Variable(tf.truncated_normal(shape=[10],mean=0,stddev=0.08))
          
         }



pred = conv_net(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_predictions = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  train_loss = []
  test_loss = []
  train_accuracy = []
  test_accuracy = []
  summary_writer = tf.summary.FileWriter('./Output',sess.graph)
  for i in range(n_iterations):
    for batch in range(len(X_train)//batch_size):
      batch_x = X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
      batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))] 
      opt = sess.run(optimizer, feed_dict={x: batch_x,y: batch_y})
      loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
    print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    print("Optimization Finished!")

    test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: X_test,y : y_test})
    train_loss.append(loss)
    test_loss.append(valid_loss)
    train_accuracy.append(acc)
    test_accuracy.append(test_acc)
    print("Testing Accuracy:","{:.5f}".format(test_acc))
  summary_writer.close()
