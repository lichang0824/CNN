import pickle
parameters_dict = {
    'kernel_size': {
        'values': [3, 4, 5]
    },
    'activation_fn': {
        'values': ['ReLU', 'Sigmoid']
    },
    'epochs_choice': {
          'values': [5, 10, 20]
    },
    'learning_rate': {
        'values': [1e-4, 1e-3, 1e-2]
    },
    'batch_size': {
        'values': [4]
    },
}
parameters_dict = {
    'kernel_size': {
        'values': [3]
    },
    'activation_fn': {
        'values': ['ReLU']
    },
    'epochs_choice': {
          'values': [5]
    },
    'learning_rate': {
        'values': [1e-4]
    },
    'batch_size': {
        'values': [4]
    },
}
with open('hyper.pkl', 'wb') as f:
	pickle.dump(parameters_dict, f)
