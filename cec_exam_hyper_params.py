hyperparameter_sets = [
    {  # Score 50
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.1,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 10000,
        "latent_dim": 8,
        "train_autoencoder": True,
        "autoencoder": "ThreeLayer",
        "num_episodes_for_autoencoder": 100,
        "epochs": 10,  # loss stops going down after 15 epochs, no need to go for 25
        "batch_size": 124,
        "use_autoencoder": True,
    },
    {  # Score 50
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.1,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 5000,
        "latent_dim": 8,
        "train_autoencoder": True,
        "autoencoder": "ThreeLayer",
        "num_episodes_for_autoencoder": 100,
        "epochs": 10,  # loss stops going down after 15 epochs, no need to go for 25
        "batch_size": 64,
        "use_autoencoder": False,
    },
    {  # 200 Score
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.1,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 5000,
        "latent_dim": 8,
        "train_autoencoder": True,
        "autoencoder": "ThreeLayer",
        "num_episodes_for_autoencoder": 100,
        "epochs": 10,  # loss stops going down after 15 epochs, no need to go for 25
        "batch_size": 64,
        "use_autoencoder": True,
    },
    {
        # 150 Score
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 1000,
        "latent_dim": 8,
        "train_autoencoder": True,
        "autoencoder": "ThreeLayer",
        "num_episodes_for_autoencoder": 25000,
        "epochs": 5,  # loss stops going down after 15 epochs, no need to go for 25
        "batch_size": 32,
        "use_autoencoder": True,
    },
    {
        # 180 Score
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 1000,
        "latent_dim": 4,
        "train_autoencoder": False,
        "autoencoder": "ThreeLayer",
        "num_episodes_for_autoencoder": 25000,
        "epochs": 15,  # loss stops going down after 15 epochs, no need to go for 25
        "batch_size": 32,
        "use_autoencoder": True,
    },
    {
        # 170 Score
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 30000,
        "latent_dim": 4,
        "train_autoencoder": True,
        "autoencoder": "ThreeLayer",
        "num_episodes_for_autoencoder": 25000,
        "epochs": 15,  # loss stops going down after 15 epochs, no need to go for 25
        "batch_size": 32,
        "use_autoencoder": True,
    },
    {
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 10000,
        "latent_dim": 4,  # 190 Score - AutoencoderThreeLayer & 25 pre-training epocs & no RL loop autoencoder training
        "train_autoencoder": True,
        "autoencoder": "ThreeLayer",
    },
    {
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 10000,
        "latent_dim": 4,  # 170 Score - AutoencoderThreeLayer & 25 pre-training epocs & no RL loop autoencoder training
        "train_autoencoder": False,
        "autoencoder": "ThreeLayer",
    },
    {
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 10000,
        "latent_dim": 8,  # 115 Score - AutoencoderThreeLayer & 25 pre-training epocs & no RL loop autoencoder training
        "train_autoencoder": False,
        "autoencoder": "ThreeLayer",
    },
    {
        "amount_of_training_to_do": 5000,
        "eval_loops": 100,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 10000,
        "latent_dim": 8,  # 120 score with AutoEncoder & 25 pre-training epocs & no RL loop autoencoder training
        "train_autoencoder": False,
        "autoencoder": "Standard",
    },
]
