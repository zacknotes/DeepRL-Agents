import tensorflow as tf
import numpy as np

#set to 1 to see reward, weights, etc..
debug = 1
user_score = 0
agent_score = 0

#THE BANDITS (the environment)

#This code has been forked from a multi-armed bandit agent
#The environment has 

#rock = 1, paper = 2, scissors = 3
bandits = [1,2,3]
num_bandits = len(bandits)
def pullBandit(bandit):

    user_action = input("Press r for rock, p for paper, s for scissors, or e to end: ")

    if user_action == 'r':
        print('You chose rock')
        if bandit == 1:
            print('Agent chose rock too')
            print('We have a TIE')
            return 0
        elif bandit == 2:
            print('Agent chose paper')
            print('You LOSE')
            return 1
        elif bandit == 3:
            print('Agent chose scissors')
            print('You WIN')
            return -1
        
    elif user_action == 'p':
        print('You chose paper')
        if bandit == 1:
            print('Agent chose rock ')
            print('You WIN')
            return -1
        elif bandit == 2:
            print('Agent chose paper too')
            print('We have a TIE')
            return 0
        elif bandit == 3:
            print('Agent chose scissors')
            print('You LOSE')
            return 1
        
    elif user_action == 's':
        print('You chose scissors')
        if bandit == 1:
            print('Agent chose rock')
            print('You LOSE')
            return 1
        elif bandit == 2:
            print('Agent chose paper')
            print('You WIN')
            return -1
        elif bandit == 3:
            print('Agent chose scissors too')
            print('We have a TIE')
            return 0

    elif user_action == 'e':
        print('You chose to end the game')
        return 2
        
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
        print ('Game #',i+1)
        #Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            #random action - a random action 0-3
            action = np.random.randint(num_bandits)
            if debug:
                print ('Random Action:',str(action))
        else:
            #chosen action - the current top pick
            action = sess.run(chosen_action)
            if debug:
                print ('Chosen Action:',str(action))


        #So, reward is either = 1 or -1
        #Action is either 0,1,2 or 3


        #Get our reward from picking one of the bandits.
        reward = pullBandit(bandits[action])
        if reward == 1:
            agent_score+=1
        elif reward == -1:
            user_score+=1
        #escape from the loop, end the game
        elif reward == 2:
            break
            
        print ('Scoreboard: YOU - ',user_score,' AGENT - ',agent_score)
    
        if debug:
            print ('Reward:',str(reward))
       

        #Update the network.
        #Remember - AGENT: weights, chosen_action, reward_holder, action_holder, responsible_weight, loss, optimizer, update
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        

        
        #Update our running tally of scores.
        total_reward[action] += reward

                
        if debug:
            print ('Running reward for the',str(num_bandits),'bandits: ',str(total_reward))
            print('Weights: ',sess.run(weights))
        print (' ');
        i+=1







