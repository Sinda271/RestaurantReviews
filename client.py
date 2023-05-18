import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils
import sys
import warnings
from fastapi import FastAPI

if not sys.warnoptions:
    warnings.simplefilter("ignore")

app = FastAPI()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, model):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    def get_parameters(self):
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):
        utils.set_model_params(self.model, parameters)
        self.model.fit(self.x_train, self.y_train)
        return utils.get_model_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        utils.set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.x_test))
        accuracy = self.model.score(self.x_test, self.y_test)
        print("accuracy: ", accuracy)
        return loss, len(self.x_test), {"accuracy": accuracy}

@app.post("/participateFL")
def listen_and_participate(train_start:int, train_end:int, ipaddress:str ,port:int):
    
    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=10,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)


    # Load dataset
    df = pd.read_csv('cleaned_dataset.csv')
    X = df.drop(['Target'], axis=1)
    y = df['Target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=False)
    x_train, y_train = x_train[train_start:train_end], y_train[train_start:train_end]


    # Start Flower client
    fl.client.start_numpy_client(
        server_address=ipaddress + ':' + str(port),
        client=FlowerClient(x_train, y_train, x_test, y_test, model),
        grpc_max_message_length=1024 * 1024 * 1024
    )


