# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:43:03 2022

@author: Utente
"""

import flwr as fl

# Start Flower server
fl.server.start_server(
#    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)