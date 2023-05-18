import flwr as fl
import numpy as np
import os
from typing import Callable, Dict
import argparse
import datetime as dt
from openstack import connection
from fastapi import FastAPI


app = FastAPI()

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, session, num_rounds):
        self.session = session
        self.num_rounds = num_rounds

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:

            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")

            if not os.path.exists(f"Session-{self.session}"):
                os.makedirs(f"Session-{self.session}")
                if rnd < self.num_rounds:
                    np.save(f"Session-{self.session}/round-{rnd}-weights.npy", aggregated_weights)
                elif rnd == self.num_rounds:
                    np.save(f"Session-{self.session}/global_session_model.npy", aggregated_weights)
            else:
                if rnd < self.num_rounds:
                    np.save(f"Session-{self.session}/round-{rnd}-weights.npy", aggregated_weights)
                elif rnd == self.num_rounds:
                    np.save(f"Session-{self.session}/global_session_model.npy", aggregated_weights)

        return aggregated_weights


    # Define batch-size, nb of epochs and verbose for fitting
    def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
        """Return a function which returns training configurations."""

        def fit_config(rnd: int) -> Dict[str, str]:
            config = {
                "batch_size": 32,
                "epochs": 50,
                "verbose": 0,
            }
            return config

        return fit_config


    # Define hyper-parameters for evaluation
    def evaluate_config(rnd: int):
        val_steps = 5 if rnd < 4 else 10
        return {"val_steps": val_steps, "verbose": 0}


@app.post("/FLsession")
def start_fl_session(num_rounds:int, ipaddress:str, port:int):

    # define date and time to save weights in directories
    today = dt.datetime.today()
    session = today.strftime("%d-%m-%Y-%H-%M-%S")

    
    # Create strategy and run server
    # Load last session weights if they exist
    sessions = ['no session']
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            if name.find('Session') != -1:
                sessions.append(name)

    if os.path.exists(f'{sessions[-1]}/global_session_model.npy'):
        initial_parameters = np.load(f"{sessions[-1]}/global_session_model.npy", allow_pickle=True)
        initial_weights = initial_parameters[0]
    else:
        initial_weights = None

    strategy = SaveModelStrategy(
        session=session,
        num_rounds=num_rounds,
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        initial_parameters=initial_weights,
    )


    fl.server.start_server(
        server_address=ipaddress + ':' + str(port),
        config={"num_rounds": num_rounds},
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )


