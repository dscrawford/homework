import project
import argparse

parser = argparse.ArgumentParser(description='Deep learning model focusing on identifying entailment')
parser.add_argument('--N', metavar='N', type=int, help='Number of epochs', nargs='?', default=10)
parser.add_argument('--L', metavar='L', type=int, help='Number of additional LSTM layers to add', nargs='?', default=0)
parser.add_argument('--max', metavar='max', type=int, help='Max length of words/embedding dimension', nargs='?', default=25)
parser.add_argument('--B', metavar='B', type=int, help='Batch size per training sample', nargs='?', default=1)
parser.add_argument('--S', metavar='S', type=float, help='Train/Test split (Just type in train split)', nargs='?', default=0.8)
parser.add_argument('--HD', metavar='D', type=int, help='Hidden dimension size', nargs='?', default=30)
parser.add_argument('--lr', metavar='lr', type=float, help='Learning rate', nargs='?', default=1e-3)
parser.add_argument('--disable_display', dest='display', help='Disables display of epochs', action='store_false')
parser.set_defaults(display=True)

args = parser.parse_args()
project.n_epochs = args.N
project.lstm_layers = args.L
project.max_length = args.max
project.embedding_dim = args.max
project.batch_size = args.B
project.train_split = args.S
project.test_split = 1.0 - args.S
project.hidden_dim = args.HD
project.lr = args.lr
project.display_epoch = args.display

project.run()



