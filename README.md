## Project Organization

```
├── LICENSE              <- Open-source license if one is chosen
├── README.md            <- The top-level README for developers using this project
├── data
│   ├── sensor.csv              <- Sensor data from Serial.serial (Arduino)
│
├── saved_models       
│   ├── '%s-e%s.keras'   <- Trained machine learning models
│
├── requirements.txt     <- The requirements file to install correct versions of libraries being used
│
└── core                        <- Source code for this project
    │
    ├── __init__.py             <- Makes core a Python module
    │
    ├── data_processor.py       <- Loads and transforms data for LSTM
    │
    ├── model.py                <- Builds, trains, and predicts with LSTM
    │
    ├── utils.py                <- Provides utility functions like timing operations
│
├── config.json          <- Defines model and data configurations
│
└── main.py              <- Main run file that executes central commands
```

--------