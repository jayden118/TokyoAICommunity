# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    def predict(self, x): 
        x = self.forward(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            #train_loss += np.sqrt(loss.data[0]*mean_corrector)
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

 
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        #test_loss += np.sqrt(loss.data[0]*mean_corrector)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

user_id = 23
user_input = Variable(test_set[user_id-1]).unsqueeze(0)
output = sae.predict(user_input)
output = output.data.numpy()
input_output = np.vstack([user_input, output])

# Recommend list of movies for specific user
from operator import itemgetter

def Prediction(user_id, nb_recommend):
    user_input = Variable(test_set[user_id - 1]).unsqueeze(0)
    predict_output = sae.predict(user_input)
    predict_output = predict_output.data.numpy()
    predicted_result = np.vstack([user_input, predict_output])
    
    trian_movie_id = np.array([i for i in range(1, nb_movies+1)]) # Create a temporary index for movies since we are going to delete some movies that the user had seen,
    recommend = np.array(predicted_result)
    recommend = np.row_stack((recommend, trian_movie_id)) # Insert that index into the result array,
    recommend = recommend.T # Transpose row and col
    recommend = recommend.tolist() # Transfer into list for further process
    
    movie_not_seen = [] # Delete the rows comtaining the movies that the user had seen
    for i in range(len(recommend)):
        if recommend[i][0] == 0.0:
            movie_not_seen.append(recommend[i])

movie_not_seen = sorted(movie_not_seen, key=itemgetter(1), reverse=True) # Sort the movies by mark

    recommend_movie = [] # Create list for recommended movies with the index we created
    for i in range(0, nb_recommend):
        recommend_movie.append(movie_not_seen[i][2])
    
    recommend_index = [] # Get the real index in the original file of 'movies.dat' by using the temporary index
    for i in range(len(recommend_movie)):
        recommend_index.append(movies[(movies.iloc[:,0]==recommend_movie[i])].index.tolist())
    
    recommend_movie_name = [] # Get a list of movie names using the real index
    for i in range(len(recommend_index)):
        np_movie = movies.iloc[recommend_index[i],1].values # Transefer to np.array
        list_movie = np_movie.tolist() # Transfer to list
        recommend_movie_name.append(list_movie)
    
    print('Highly Recommended Moives for You:\n')
    for i in range(len(recommend_movie_name)):
        print(str(recommend_movie_name[i]))
    
    return recommend_movie_name

user_id = 367
nb_recommend = 20
movie_for_you = Prediction(user_id = user_id, nb_recommend = nb_recommend)
