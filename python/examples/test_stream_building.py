from avalanche.benchmarks.generators import tensors_benchmark
import torch

pattern_shape = (3, 32, 32)

# Definition of training experiences
# Experience 1
experience_1_x = torch.zeros(100, *pattern_shape)
experience_1_y = torch.zeros(100, dtype=torch.long)

# Experience 2
experience_2_x = torch.zeros(80, *pattern_shape)
experience_2_y = torch.ones(80, dtype=torch.long)

# Test experience
# For this example we define a single test experience,
# but "tensors_benchmark" allows you to define even more than one!
test_x = torch.zeros(50, *pattern_shape)
test_y = torch.zeros(50, dtype=torch.long)

generic_scenario = tensors_benchmark(
    train_tensors=[(experience_1_x, experience_1_y), (experience_2_x, experience_2_y)],
    test_tensors=[(test_x, test_y)],
    task_labels=[0, 0],  # Task label of each train exp
    complete_test_set_only=True
)