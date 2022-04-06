import subprocess
import numpy as np


# Change the parameters here to evaluate the model's test metrics
model = 'VGNAE'
dataset = 'Cora'
epochs = 300
output_channels = 128
training_rate = 0.8
lr = 0.005

NUM_TIMES = 10

auc_list = list()
ap_list = list()

for ii in range(NUM_TIMES):
    result = subprocess.run(['python', 'main_a1.py', '--model', f'{model}', '--dataset',
                             f'{dataset}', '--epochs', f'{epochs}', '--channels', f'{output_channels}',
                             '--training_rate', f'{training_rate}', '--learning_rate', f'{lr}'],
                            stdout=subprocess.PIPE)

    terminal_output = result.stdout.decode('utf-8')
    terminal_output = [x.strip() for x in terminal_output.split('\n')]

    if 'TEST' not in terminal_output[-1]:
        terminal_output = terminal_output[:-1]

    test_op = terminal_output[-3], terminal_output[-1]

    test_metrics = [[float(y.split(':')[-1]) for y in x.split(',')] for x in test_op]

    best_auc = max(test_metrics[0][0], test_metrics[1][0])
    best_ap = max(test_metrics[0][1], test_metrics[1][1])

    print(f'Model {ii} trained')
    print('Test AUC:', best_auc)
    print('Test AP:', best_ap)
    print()

    auc_list.append(best_auc)
    ap_list.append(best_ap)

print('Loop finished!')
print(f'Model: {model}, Dataset: {dataset}')
print(f'Average test AUC: {np.mean(auc_list):.5f} +- {np.std(auc_list):.5f}')
print(f'Average test AP: {np.mean(ap_list):.5f} +- {np.std(ap_list):.5f}')

print('Done')
