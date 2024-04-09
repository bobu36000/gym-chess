import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(epoch_rewards, lr, discount, epsilon, goal=100):
    x = np.linspace(1, len(epoch_rewards), len(epoch_rewards))
    y = np.array(epoch_rewards)

    plt.plot(x, epoch_rewards)
    plt.hlines(goal, 0, len(epoch_rewards)-1, color="b", linestyles="--")
    plt.title("Reward over the epochs of training")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.text(1,95, f"lr={lr}")
    plt.text(1,90, f"discount={discount}")
    plt.text(1,85, f"epsilon={epsilon}")
    plt.grid(True)
    plt.show()

def plot_test_rewards(epoch_rewards, test_rewards, goal=100):
    x = np.linspace(0, len(epoch_rewards), len(epoch_rewards)+1)

    epoch_rewards.insert(0, 0)
    test_rewards.insert(0, 0)

    plt.plot(x, epoch_rewards)
    plt.plot(x, test_rewards)
    plt.hlines(goal, 0, len(epoch_rewards)-1, color="b", linestyles="--")

    plt.title("Reward over the epochs of training")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.ylim(-20, 120)
    plt.grid(True)
    plt.show()

def plot_episode_lengths(train_lengths, test_lengths):
    x = np.linspace(1, len(train_lengths)-1, len(train_lengths))

    plt.plot(x, train_lengths)
    plt.plot(x, test_lengths)
    
    plt.title("Episode lengths over the epochs of training")
    plt.xlabel("Epoch")
    plt.ylabel("Episode Length")
    plt.grid(True)
    plt.show()
