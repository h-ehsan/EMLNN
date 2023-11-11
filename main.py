import argparse
import utils
import numpy as np
import model


def main():
    parser = argparse.ArgumentParser(description='Run an experiment with EMLNN.')
    parser.add_argument('train_data', type=str, help='Path to the training dataset file')
    parser.add_argument('test_data', type=str, help='Path to the testing dataset file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=(256, 256, 128, 64),
                        help='Hidden layer sizes')
    parser.add_argument('--dropout_ratio', type=float, default=0.4, help='Dropout ratio')

    args = parser.parse_args()

    # import train and test dataset and remove headers
    x_train, train_label_sets = utils.load_dataset(args.train_data, 2)
    x_test, test_label_sets = utils.load_dataset(args.test_data, 2)

    # normalize the dataset
    x_train, x_test = utils.normalize(x_train, x_test, method='minmax')

    # shuffle the dataset
    x_train, train_label_sets = utils.shuffle(x_train, train_label_sets)

    # Estimate the input size by looking at the number of columns in train and test.
    input_size = x_train.shape[1]

    # Estimate the number of labels in each label set by looking at the number of unique values in each label set.
    num_classes = [len(np.unique(train_label_sets[0])), len(np.unique(train_label_sets[1]))]

    IDS = model.emlnn(input_size, num_classes,
                      batch_size=args.batch_size,
                      learning_rate=args.learning_rate,
                      epochs=args.epochs,
                      hidden_layer_sizes=args.hidden_layer_sizes,
                      dropout_ratio=args.dropout_ratio)

    IDS.train(x_train, train_label_sets)

    predictions = IDS.predict(x_test)

    print("EMR:", utils.exact_match_ratio(predictions, test_label_sets))


if __name__ == '__main__':
    main()
