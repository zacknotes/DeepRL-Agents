import tensorflow as tf
import numpy as np

#set to 1 to see reward, weights, etc..
debug = 1

user_score = 0
agent_score = 0

#THE CHOICES 
#rock = 1, paper = 2, scissors = 3
choices = [1,2,3]

num_choices = len(choices)

print(' ')
print('Lets play rocks, paper, scissors!')
print('First one to score 10 wins.')
print(' ')
      
def match(choices):

    user_action = input("Press r for rock, p for paper, s for scissors, or e to end: ")

    if user_action == 'r':
        print('You chose ROCK')
        if choices == 1:
            print('Agent chose ROCK too')
            print('We have a TIE')
            return 0
        elif choices == 2:
            print('Agent chose PAPER')
            print('You LOSE')
            return 1
        elif choices == 3:
            print('Agent chose SCISSORS')
            print('You WIN')
            return -1
        
    elif user_action == 'p':
        print('You chose PAPER')
        if choices == 1:
            print('Agent chose ROCK ')
            print('You WIN')
            return -1
        elif choices == 2:
            print('Agent chose PAPER too')
            print('We have a TIE')
            return 0
        elif choices == 3:
            print('Agent chose SCISSORS')
            print('You LOSE')
            return 1
        
    elif user_action == 's':
        print('You chose SCISSORS')
        if choices == 1:
            print('Agent chose ROCK')
            print('You LOSE')
            return 1
        elif choices == 2:
            print('Agent chose PAPER')
            print('You WIN')
            return -1
        elif choices == 3:
            print('Agent chose SCISSORS too')
            print('We have a TIE')
            return 0

    elif user_action == 'e':
        print('You chose to end the game')
        return 2
        
    else:
        print('Try again')
        return 0 
        

    #fake breakpoint
    #user_action = input("This is a fake breakpoint. Press enter!")


#THE AGENT
#The code below established our simple neural agent.
#It consists of a set of values for each of the choices.
#Each value is an estimate of the value of the return from the various choices.
#We use a policy gradient method to update the agent by moving the value for the selected action toward the recieved reward.

tf.reset_default_graph()

#These two lines established the feed-forward part of the network. This does the actual choosing.

weights = tf.Variable(tf.ones([num_choices]))
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
total_reward = np.zeros(num_choices) #Set scoreboard for all four choices to 0.
e = 0.6 #Set the chance of taking a random action.


#may need to replace with tf.global_variables_initializer
#init = tf.initialize_all_variables() # old init method
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    i = 0
    maxScore = 0

    #at this point total reward = [0,0,0,0]
    while maxScore < 10:
        print ('Game #',i+1)
        i+=1
        #This is where the agent decides whether to throw rocks, paper or scissors
        #It chooses either a random action or the one from our network that has the highest weight
        if np.random.rand(1) < e:
            #random action - a random action 0-2
            action = np.random.randint(num_choices)
            if debug:
                print ('Random Action:',str(action))
        else:
            #chosen action - the current top pick
            action = sess.run(chosen_action)
            if debug:
                print ('Chosen Action:',str(action))


        #So, reward is either = 1, 0, or -1
        #Action is either 0,1,2 (rocks, paper, scissors)


        #This is where the user enters their action (rocks, paper, or scissors)
        reward = match(choices[action])
        if reward == 1:
            agent_score+=1
        elif reward == -1:
            user_score+=1
        #escape from the loop, end the game
        elif reward == 2:
            break

        print ('   ')
        print ('*******************************************')
        print ('Scoreboard: YOU - ',user_score,' AGENT - ',agent_score)
        print ('*******************************************')
        print ('   ')
    
        if debug:
            print ('Reward:',str(reward))
       

        #Update the network.
        #Remember - AGENT: weights, chosen_action, reward_holder, action_holder, responsible_weight, loss, optimizer, update
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        

        
        #Update our running tally of scores.
        total_reward[action] += reward

                
        if debug:
            print ('Running reward for the',str(num_choices),'choices: ',str(total_reward))
            print('Weights: ',sess.run(weights))
        print (' ');
        maxScore = max(user_score,agent_score)

if user_score > agent_score:
     print('GAME OVER, YOU WIN!')
else:
   print('GAME OVER, YOU LOSE!')
    
        







