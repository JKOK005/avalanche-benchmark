from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.generators import tensors_benchmark
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from models.VGG16 import Vgg16
import argparse
import glob
import os
import torch

"""
python3 pipeline/vanilla.py \
--net vgg \
--train_path /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/train/ \
--test_path /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/test/
"""

torch.manual_seed(0)

def get_vgg_net():
	return Vgg16(num_classes = 10)

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='Vanilla model training using Avalanche')
	parser.add_argument('--net', type = str, nargs = '?', help = 'Type of network')
	parser.add_argument('--train_path', type = str, nargs = '?', help = 'Path to training CORe50 dataset')
	parser.add_argument('--test_path', type = str, nargs = '?', help = 'Path to evaluation CORe50 dataset')
	args 		= parser.parse_args()

	model 		= get_vgg_net() if args.net == "vgg" else None
	optimizer 	= Adam(model.parameters(), lr = 5e-5)
	objective 	= CrossEntropyLoss()

	strategy 	= Naive(
				    model, optimizer, objective,
				    train_mb_size = 32, train_epochs = 30, eval_mb_size = 32,
				)

	results = []

	all_test_X 	= []
	all_test_Y 	= []

	for each_file in sorted(glob.glob(f"{args.test_dir}/*.npy")):
		data 	= np.load(each_file, allow_pickle = True)
		all_test_X.append(np.array(data[:, 0].tolist()))
		all_test_Y.append(np.array(data[:, 1].tolist()))

	test_X 	= np.concatenate(all_test_X)
	test_Y 	= np.concatenate(all_test_Y)

	test_X 	= torch.from_numpy(test_X)
	test_Y 	= torch.from_numpy(test_Y)

	for each_file in sorted(glob.glob(f"{args.train_dir}/*.npy")):
		data 	= np.load(each_file, allow_pickle = True)
		train_X = np.array(data[:, 0].tolist())
		train_Y = np.array(data[:, 1].tolist())

		train_X = torch.from_numpy(train_X)
		train_Y = torch.from_numpy(train_Y)

		generic_scenario = tensors_benchmark(
			train_tensors	= [(train_X, train_Y)],
			test_tensors	= [(test_X, test_Y)],
			task_labels		= [0],  # Task label of each train exp
			complete_test_set_only = True
		)

		for experience in generic_scenario.train_stream:
			cl_strategy.train(experience)
			results.append(cl_strategy.eval(generic_scenario.test_stream))

	print(results)