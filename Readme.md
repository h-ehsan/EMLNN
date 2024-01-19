# Ensemble-based Multi-Label Neural Network (EMLNN)

This is an implementation of the EMLNN algorithm proposed in the following paper:

[*Expanding Analytical Capabilities in Intrusion Detection through Ensemble-Based Multi-Label Classification*, Computers & Security, pp. 103730, 2024.](https://doi.org/10.1016/j.cose.2024.103730)


**Instructions:**

Run the code using the following command (should navigate to the code folder first):
python main.py [address to local train data.csv] [address to local test data.csv]

Optional: Adjust parameters as needed:<br>
--epochs [value] --batch_size [value] --learning_rate [value] --hidden_layer_sizes [value, value, value...] --dropout [value]

Example:<br>
python main.py train_data.csv test_data.csv --epochs 200 --batch_size 32 --learning_rate 0.001 --hidden_layer_sizes 512, 256 --dropout 0.3

**Datasets:** [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and [Bot-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset)

**Library versions:**<br>
pandas                        1.4.0<br>
scikit-learn                  1.3.2<br>
scipy                         1.9.3<br>
tensorflow                    2.14.0<br>
keras                         2.14.0<br>
numpy                         1.25.2<br>

Tested on Python 3.9.16
