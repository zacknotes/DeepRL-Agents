import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

#ACTIONS

class contextual_bandit():
    def __init__(self):

        self.state = 0

        #List out our actions

        self.bandits = np.array([

        #desktop

        #Results count for the first set,
        [12,16,20,24,28,32,36,40,44,48],

        #Bloomreach results per interest,
        [40,45,50,55,60,65,70,75,80,85],

        #Baynote recs pulled on each star,
        [8,9,10,11,12,13,14,15,16,17],

        #Items in set 2,
        [12,16,20,24,28,32,36,40,44,48],

        #mobile

        #Results count for the first set,
        [12,16,20,24,28,32,36,40,44,48],

        #Bloomreach results per interest,
        [40,45,50,55,60,65,70,75,80,85],

        #Baynote recs pulled on each star,
        [8,9,10,11,12,13,14,15,16,17],

        #Items in set 2,
        [12,16,20,24,28,32,36,40,44,48]

        ])

        self.bandit_description = [
            '[D] Results count for the first set',
            '[D] Bloomreach results per interest',
            '[D] Baynote recs pulled on each star',
            '[D] Items in set 2',
            '[M] Results count for the first set',
            '[M] Bloomreach results per interest',
            '[M] Baynote recs pulled on each star',
            '[M] Items in set 2'
            ]


        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getState(self):
        #IRL this would read the user's device type

        print('   ')
        #Returns a random state for each episode.
        #self.state = np.random.randint(0,len(self.bandits))
        self.state = np.random.randint(0,2)
        if self.state == 0:
            print('State is desktop')
        elif self.state == 1:
            print('State is mobile')
        return self.state

    def pullArm(self,action):
        #IRL this would run on every sunny session. The reward would either be


        #The state in this function is baked in to the self object
        #print('state in pullArm: ',self.state)

        #Action is the index in the array

        bandit = self.bandits[self.state,action]

        #desktop
        if self.state==0:
            star_useage = 0.2

        #mobile
        elif self.state==1:
            star_useage = 0.8

        #We simulate whether the user starred or not with a randomn number
        random_result = np.random.rand(1)


        #IRL we would see if the user starred or not
        if random_result < star_useage:
            #return a positive reward.
            print('User clicked star')
            return 1
        else:
            #return a negative reward.
            print('User did not click star')
            return -1

class agent():

    #lr = learning rate, s_size = state size, a_size = action size
    def __init__(self, lr, s_size,a_size):

        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[1],dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in,s_size)
        output = slim.fully_connected(state_in_OH,a_size,\
            biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output,self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)

        #these were needed to run experience traces
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        #here is how we updated before using experience traces
        #self.update = optimizer.minimize(self.loss)

        #this is where we tell agent to update with experience traces
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))



#TRAINING THE AGENT
#We will train our agent by taking actions in our environment, and recieving rewards.
#Using the rewards and actions, we can know how to properly update our network
#in order to more often choose actions that will yield the highest rewards over time.


tf.reset_default_graph() #Clear the Tensorflow graph.

cBandit = contextual_bandit() #Load the bandits.

#Load the agent.
myAgent = agent(lr=0.001,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
print('Waking up the elephant...',cBandit.num_bandits)

#The weights we will evaluate to look into the network.
weights = tf.trainable_variables()[0]

total_episodes = 100 #Set total number of episodes to train agent on.
total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
e = 0.9 #Set the chance of taking a random action.
update_frequency = 5 #This is how often we update with experience traces

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()


#####MAIN######

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)

    #Create a saver object which will save all the variables
    saver = tf.train.Saver()

    #Comment this out on first run
    try:
        saver.restore(sess, "/tmp/my-model")
    except tf.errors.NotFoundError:
        print('tf.errors.NotFoundError: \n this happens on first run of model')
    i = 0

    #need to clean this up a bit
    action=[0,0,0,0,0,0,0,0]

    print('Initial Weights: ',sess.run(weights))
    #fake breakpoint
    user_action = input("Finished reviewing the weights? Press enter!")

    #This is where we initialize the buffer for experience traces
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    #Each 'episode' is another Sunny Session
    for i in range(total_episodes):

        #Get a state for the environment.
        s = cBandit.getState()

        #cycles through actions #0-3 if desktop (0)and actions #4-7 if mobile (1)
        for j in range(4*s,cBandit.num_bandits - (4*(1-s))):

            print('j = ',j)

            #Choose either a random action or one from our network of weights

            if np.random.rand(1) < e:
                #random action - this is the index into the array
                action[j] = np.random.randint(cBandit.num_actions)
                print('Random Action',str(action))

            else:
                #chosen action
                action[j] = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[j]})
                print('Chosen Action',str(action))

        #Get our reward for taking an action given a bandit.
        reward = cBandit.pullArm(action)


        #We want to update so that we don't do this after every sunny session (episode)
        #Instead we want to save to a buffer and run experience traces

        #Update the network
        #cycles through updates on actions #0-3 if desktop (s=0)and actions #4-7 if mobile (s=1)
        for j in range(4*s,cBandit.num_bandits - (4*(1-s))):

            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action[j]],myAgent.state_in:[j]}

            #this is the line we had before we implemented experience traces
            #_,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

            #experience trace updates
            grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
            for idx,grad in enumerate(grads):
                gradBuffer[idx] += grad

            if i % update_frequency == 0 and i != 0:
                print('updating weights with experience trace. Action = ',j)
                feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)

                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

            #Update our running tally of scores.
            total_reward[j,action[j]] += reward


    #fake breakpoint
    user_action = input("Wanna see the weights now? Press enter!")
    print ('Mean reward for each of the',str(cBandit.num_bandits),' bandits: ',str(np.mean(total_reward,axis=1)))
    print('Weights: ',sess.run(weights))


    #fake breakpoint
    user_action = input("Done reviewing the weights? Press enter and I'll save the weights")

    #Now, save the graph
    saver.save(sess, "/tmp/my-model")
    #print ('Reward',str(total_reward))

    # to pull the weights into an np array
    # I'm using the TF session here and saving them as ww
    # ww was the variable name I used in a previous version of the code
    ww = sess.run(weights)


for bandit_index in range(int(cBandit.num_bandits / 2)):
    print ('On desktop, the agent thinks action',str(np.argmax(ww[bandit_index])+1),' for bandit',str(bandit_index+1),' is the most promising....')


    best_action_index = ww[bandit_index].argmax()
    print("{}:\t{}".format(
    cBandit.bandit_description[bandit_index],
    cBandit.bandits[bandit_index][best_action_index]
     )
    )


for bandit_index in range(int(cBandit.num_bandits / 2), cBandit.num_bandits):
    print ('On mobile, the agent thinks action',str(np.argmax(ww[bandit_index])+1),' for bandit',str(bandit_index+1),' is the most promising....')

    best_action_index = ww[bandit_index].argmax()
    print("{}:\t{}".format(
    cBandit.bandit_description[bandit_index],
    cBandit.bandits[bandit_index][best_action_index]
     )
    )

