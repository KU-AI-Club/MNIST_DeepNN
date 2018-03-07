import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

def add_layers(nodes_per_lay,num_lay,lay_1):
	w = tf.Variable(tf.random_uniform([nodes_per_lay,nodes_per_lay]))
	b = tf.Variable(tf.random_uniform([nodes_per_lay]))
	y = tf.nn.relu(tf.matmul(lay_1,w)+b)
	if num_lay == 0:
		return y
	else:
		return add_layers(nodes_per_lay,num_lay-1,y)


batch_size = 100
num_classes = 10
num_steps = 2000
num_layers = 2
nodes_per_lay = 10
epochs = 20

y_sen = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,[None,num_classes])


W_in = tf.Variable(tf.truncated_normal([784,nodes_per_lay],stddev=.1))
b_in = tf.Variable(tf.truncated_normal([nodes_per_lay],stddev=.1))
y_in_init = tf.matmul(x,W_in) + b_in
y_in = tf.nn.relu(y_in_init)

y_hid = add_layers(nodes_per_lay,num_layers,y_in)

W_out = tf.Variable(tf.truncated_normal([nodes_per_lay,num_classes],stddev=.1))
b_out = tf.Variable(tf.truncated_normal([num_classes],stddev=.1))
y = tf.matmul(y_hid,W_out) + b_out

#Step 4) Loss Function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#Step 5) Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.03).minimize(cost)

#Step 6) Create Session

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for ep in range(epochs):
		for steps in range(1000):
			batch_x,batch_y = mnist.train.next_batch(100)
			sess.run(optimizer,feed_dict={x:batch_x,y_true:batch_y})

		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))

		acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
		print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))



