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
parameters_dict_list = []
for kernel_size in [3, 5]:
    for activation_fn in ['ReLU']:
        for epochs_choice in [5, 10, 15]:
            for learning_rate in [1e-4, 1e-5]:
                for batch_size in [4]:
                    parameters_dict = {
                        'kernel_size': {
                            'values': [kernel_size]
                        },
                        'activation_fn': {
                            'values': [activation_fn]
                        },
                        'epochs_choice': {
                              'values': [epochs_choice]
                        },
                        'learning_rate': {
                            'values': [learning_rate]
                        },
                        'batch_size': {
                            'values': [batch_size]
                        },
                    }
                    parameters_dict_list.append(parameters_dict)
with open('hyper.pkl', 'wb') as f:
	pickle.dump(parameters_dict_list, f, pickle.HIGHEST_PROTOCOL)
