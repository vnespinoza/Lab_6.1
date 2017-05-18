import os, numpy, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def skipgram(line, vocab, ws):
	"""Convert line into skip-gram data points."""
	inputs = []; labels = []
	word_list = line.split()
	for i in range( len(word_list) ):
		input_word = word_list[i]
		left = word_list[max(i-ws,0) : i]
		right = word_list[i+1 : i+1+ws]
		for context_word in left+right:
			if (input_word in vocab) and (context_word in vocab):
				inputs.append( vocab.index(input_word) )
				labels.append( vocab.index(context_word) )
	labels = numpy.array([labels]).T # store in a column matrix	
	return inputs, labels

def cossim(x,y):
	num = numpy.dot(x,y)
	den = numpy.linalg.norm(x)*numpy.linalg.norm(y)
	return num/den

if __name__ == '__main__':
	corpus = ['roses are red', 'violets are blue']
	vocab = ['roses', 'violets', 'red', 'blue', 'are']
	vs = len(vocab) # vocabulary size
	es = 2 # embedding size
	ws = 1 # window size for skipgram

	W_hidden = tf.Variable( tf.random_normal([vs, es]) )
	W_out = tf.Variable( tf.random_normal([vs, es]) )
	b_out = tf.Variable( tf.random_normal([vs]) )

	inputs = tf.placeholder(tf.int32, [None])
	embed = tf.nn.embedding_lookup(W_hidden, inputs)
	labels = tf.placeholder(tf.int32, [None, 1])
	loss = tf.nn.sampled_softmax_loss(W_out, b_out, labels, embed, 3, vs, 1)
	loss = tf.reduce_mean(loss)
	train = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

	sess = tf.Session()
	# 1. Train the network.
	sess.run( tf.global_variables_initializer() )
	for i in range(100):
		for line in corpus:
			xs, ys = skipgram(line, vocab, ws)
			sess.run(train, feed_dict={inputs:xs, labels:ys})
	# 2. Print out the word vectors.
	vectors = sess.run(embed, feed_dict={inputs:range(vs)})
	for i in range( len(vocab) ):
		print vocab[i], vectors[i]
	# 3. Done.
	sess.close()
	
	roses = vectors[vocab.index('roses')]
	red = vectors[vocab.index('red')]
	violets = vectors[vocab.index('violets')]
	target = red - roses + violets

	scores = []
	for i in range(vs):
		scores.append(cossim(vectors[i], target)) #calculates cosine similarity 
	scores = numpy.array(scores) #save scores in numpy array
	x = vocab[scores.argmax()]
	print  x

	
	
	
