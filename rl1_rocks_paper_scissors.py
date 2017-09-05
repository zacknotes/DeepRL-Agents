import tensorflow as tf
import numpy as np


#THE BANDITS (the environment)

#Here we define our bandits. For this example we are using a four-armed bandit.
#The pullBandit function generates a random number from a normal distribution with a mean of 0.
#The lower the bandit number,the more likely a positive reward will be returned.
#We want our agent to learn to always choose the bandit that will give that positive reward.
#Currently bandit 4 (index#3) is set to most often provide a positive reward.

#rock = 1, paper = 2, scissors = 3
bandits = [1,2,3]
num_bandits = len(bandits)
def pullBandit(bandit):

    user_action = input("Your turn! (press r for rock, p for paper, or  s for scissors) ")

    if user_action == 'r':
        print('You chose rock')
        if bandit == 1:
            print('Agent chose rock too')
            return 0
        elif bandit == 2:
            print('Agent chose paper')
            return 1
        elif bandit == 3:
            print('Agent chose scissors')
            return -1
        
    elif user_action == 'p':
        print('You chose paper')
        if bandit == 1:
            print('Agent chose rock ')
            return -1
        elif bandit == 2:
            print('Agent chose paper too')
            return 0
        elif bandit == 3:
            print('Agent chose scissors')
            return 1
        
    elif user_action == 's':
        print('You chose scissors')
        if bandit == 1:
            print('Agent chose rock')
            return 1
        elif bandit == 2:
            print('Agent chose paper')
            return -1
        elif bandit == 3:
            print('Agent chose scissors too')
            return 0
        
    else:
        print('Try again')
        

    #fake breakpoint
    user_action = input("This is a fake breakpoint. Press enter!")
    
    return 1


#THE AGENT
#The code below established our simple neural agent.
#It consists of a set of values for each of the bandits.
#Each value is an estimate of the value of the return from choosing the bandit.
#We use a policy gradient method to update the agent by moving the value for the selected action toward the recieved reward.

tf.reset_default_graph()

#These two lines established the feed-forward part of the network. This does the actual choosing.

weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights,0)


#The next six lines establish the training procedure
#We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)




#TRAINING THE AGENT
#We will train our agent by taking actions in our environment, and recieving rewards.
#Using the rewards and actions, we can know how to properly update our network
#in order to more often choose actions that will yield the highest rewards over time.

#total_episodes = 1000 #Set total number of episodes to train agent on.
total_episodes = 100 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits) #Set scoreboard for all four bandits to 0.
e = 0.6 #Set the chance of taking a random action.


#may need to replace with tf.global_variables_initializer
#init = tf.initialize_all_variables() # old init method
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    i = 0

    #at this point total reward = [0,0,0,0]
    while i < total_episodes:
        print ('Episode #:',i+1)
        #Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            #random action - a random action 0-3
            action = np.random.randint(num_bandits)
            print ('Random Action:',str(action))
        else:
            #chosen action - the current top pick
            action = sess.run(chosen_action)
            print ('Chosen Action:',str(action))


        #So, reward is either = 1 or -1
        #Action is either 0,1,2 or 3


        #Get our reward from picking one of the bandits.
        reward = pullBandit(bandits[action]) 
    
        print ('Reward:',str(reward))
       

        #Update the network.
        #Remember - AGENT: weights, chosen_action, reward_holder, action_holder, responsible_weight, loss, optimizer, update
        print ('Updating the network...')
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        

        
        #Update our running tally of scores.
        total_reward[action] += reward

        #if i % 50 == 0:
        #   print ('Running reward for the',str(num_bandits),'bandits: ',str(total_reward))
                
        print ('Running reward for the',str(num_bandits),'bandits: ',str(total_reward))
        print('Weights: ',sess.run(weights))
        print (' ');
        i+=1

print ('The agent thinks bandit ',str(np.argmax(ww)+1),' is the most promising....')
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print ('...and it was right!')
else:
    print ('...and it was wrong!')





