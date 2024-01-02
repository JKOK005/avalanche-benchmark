from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators import tensors_benchmark
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, LwFPlugin
import torch
from models.VGG16 import Vgg16

torch.manual_seed(0)

model = SimpleMLP(num_classes = 10)
# model = Vgg16(num_classes = 10)

optimizer = SGD(model.parameters(), lr = 0.001, momentum=0.9)

criterion = CrossEntropyLoss()

ewc = EWCPlugin(ewc_lambda = 0.001)

lwf = LwFPlugin()

cl_strategy = Naive(
    model, optimizer, criterion,
    train_mb_size = 100, train_epochs = 1, eval_mb_size = 100,
    plugins = [lwf]
)

# experience_1_x = torch.zeros(100, 3, 128, 128)
# experience_1_y = torch.zeros(100, dtype = torch.long)

# test_x = torch.zeros(50, 3, 128, 128)
# test_y = torch.zeros(50, dtype = torch.long)

# generic_scenario = tensors_benchmark(
#     train_tensors=[(experience_1_x, experience_1_y)],
#     test_tensors=[(test_x, test_y)],
#     task_labels=[0],  # Task label of each train exp
#     complete_test_set_only=True
# )

generic_scenario = SplitMNIST(n_experiences = 1, seed = 1)

# TRAINING LOOP
print('Starting experiment...')
results = []

for _ in range(3):
    for experience in generic_scenario.train_stream:
        cl_strategy.train(experience)
        print('Training completed')

        # print('Computing accuracy on the whole test set')
        # results.append(cl_strategy.eval(generic_scenario.test_stream))    
    print(results)