import sys
import numpy as np
import tensorflow as tf
import math
import os

import tensorflow.contrib as contribe

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#init class

class Complex:
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def __str__(self):
        return "Complex (%f,%f)" % (self.a,self.b)

    def __add__(self,other):
        return Complex(self.a + other.a, self.b + other.b)

complex1 = Complex(3.0,-4.5)
complex2 = Complex(4.0,20.1)
print(complex1.a,complex1.b)
print(complex1 + complex2)

#graph eval

aa = tf.constant(1)
bb = tf.constant(2)
cc = tf.constant(3)
dd = tf.constant(4)

add1 = tf.add(aa,bb)
mul1 = tf.div(bb,cc)
add2 = tf.multiply(bb,cc)
output = tf.add(add2,mul1)


bb1 = tf.Variable(tf.zeros(10))

#matrix add

ee = tf.Variable(tf.zeros([3,2]))
dd = tf.Variable(tf.ones([2]))

gg = ee + dd
# sess execute

with tf.Session() as sess:
    print(sess.run(output))
    tf.global_variables_initializer().run()
    result = sess.run([ee,dd,gg])
    print(result)


#numpy
b = np.array([1,2,3,4])
e = np.eye(5,5)
print(b[0])
print(b.shape[0])
print(e.shape[1])
print(b.shape)

print("seed....")
seed = 100
seed1 = 200
seed2 = 300
print(seed if seed1 == 100 else seed2)
print(np.random.seed())

print("for in xrange")
for i in range(3):
    print(i)
print("dtype")
dt = np.array(b,dtype = np.complex)

print(dt)
print(dt.shape)
print(dt.base)
print(dt.flags)
print(bb)


# tensorflow 1.3
print("tensorflow 1.3 ................")
print(tf.compat.as_str_any("1000"))
a2 = np.array([1,2,3,4])
d2 = np.reshape(a2,(2,2))
c2 = np.reshape(a2,(1,2,2))
print(d2)
print(c2)
print(math.ceil(100))

#tf.app.flags
print("flages....................")
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_data_path","/train.txt","traning data dir")
print(FLAGS.train_data_path)


#graph

cc1 = tf.constant(4.0)
print("graph.......................")
print(cc1.graph is tf.get_default_graph())
g1 = tf.Graph()
with g1.as_default():
    cc2 = tf.constant(5.0)
    print(cc2.graph is g1)

#xrange
print("xrange..................")

for i in range(1,3):
    print(i)


#strided_slice
data = [1,2,3,4,5,6,7,8]
x = tf.strided_slice(data,[0],[4])
y = tf.strided_slice(data,[1],[5])

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))

#tf.name_scope
c_0 = tf.constant(0,name = "c")
c_1 = tf.constant(2,name = "c")

with tf.name_scope("outer"):
    c_2 = tf.constant(2, name = "c")

print("tf.name_scope................")
print(c_0)
print(c_1)
print(c_2)
#return multiple value

def more(x,y,step,angle):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)

    return nx,ny

print("return multiple value.................")
print(more(100,200,10,20))

#parameter name

def f(a,b,c):
    return a * b * c

print("parameter name................")
print(f(100,c = 10, b = 23))



#subgraph

print("subgraph..................")

x = tf.constant([[37.0,-23.0],[1.0,4.0]])
w = tf.Variable(tf.random_uniform([2,2]))
y = tf.matmul(x,w)

output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
    sess.run(init_op)

    print(sess.run(output))

    y_val,output_val = sess.run([y,output])


    print(y_val,output_val)

#feed

print("feed...............")
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:

    print(sess.run(y, {x: [1.0,2.0,3.0]}))
    print(sess.run(y, {x: [0.0,0.0,5.0]}))


#trace information

print("trace information............")
y = tf.matmul([[37.0,-23.0], [ 1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE


    metadata = tf.RunMetadata()

    sess.run(y, options = options, run_metadata=metadata)

    #print(metadata.partition_graphs)

    #print(metadata.step_stats)


#multiple graph

print("multiple graph ..............")

g_1 = tf.Graph()

with g_1.as_default():
    c = tf.constant("Node in g_1")

    sess_1 = tf.Session()

g_2 = tf.Graph()

with g_2.as_default():

    d = tf.constant("Node in g_2")

sess_2 = tf.Session(graph=g_2)

print(c.graph is g_1)
print(sess_1.graph is g_1)

print( d.graph is g_2)
print( sess_2.graph is g_2)

#saving and restoring

print("Saving and restoring ..................")
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    inc_v1.op.run()
    dec_v2.op.run()

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)


saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess,"/tmp/model.ckpt")

    print("model restored")

    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())

#importing data

print("importing data ...............")

dataset1 = contribe.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
print(dataset1.output_types)
print(dataset1.output_shapes)

dataset2 = contribe.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
     tf.random_uniform([4,100], maxval=100, dtype=tf.int32)))

print(dataset2.output_types)
print(dataset2.output_shapes)

dataset = contribe.data.Dataset.range(5)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(5):
        value = sess.run(next_element)
        print(i == value)


max_value = tf.placeholder(tf.int64, shape=[])
dataset = contribe.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={max_value : 2})
    for i in range(2):
        value = sess.run(next_element)
        print(i == value)


inc_dataset = contribe.data.Dataset.range(100)
dec_dataset = contribe.data.Dataset.range(0, -100, -1)
dataset = contribe.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))

#queue

print("threading and queues ..............")

def simple_shuffle_batch(source, capacity, batch_size = 10):
    queue = tf.RandomShuffleQueue(capacity=capacity,
                                  min_after_dequeue=int(0.9 * capacity),
                                  shapes=source.shape,
                                  dtypes=source.dtype)
    enqueue = queue.enqueue(source)

    num_threads = 4
    qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)
    tf.train.add_queue_runner(qr)

    return queue.dequeue_many(batch_size)


input = tf.constant(list(range(100)))
input = contribe.data.Dataset.from_tensor_slices(input)
input = input.make_one_shot_iterator().get_next()

get_batch = simple_shuffle_batch(input, capacity = 20)

with tf.train.MonitoredSession() as sess:

    while not sess.should_stop():
        print(sess.run(get_batch))

















