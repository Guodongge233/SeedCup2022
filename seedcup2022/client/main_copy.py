import json
import socket
from base import *
from req import *
from resp import *
from config import config
from ui import UI
import subprocess
import logging
import re
from threading import Thread
from itertools import cycle
from time import sleep
import sys

# import torch
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import time, datetime
import matplotlib.pyplot as plt

global state_origin
global score
global map_info
# logger config
logging.basicConfig(
    # uncomment this will redirect log to file *client.log*
    # filename="client.log",
    format="[%(asctime)s][%(levelname)s] %(message)s",
    filemode="a+",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
mapSize = config.get("MapSize")

# record the context of global data
gContext = {
    "playerID": None,
    "characterID": [],
    "gameOverFlag": False,
    "prompt": (
        "Take actions!\n"
        "'s': move in current direction\n"
        "'w': turn up\n"
        "'e': turn up right\n"
        "'d': turn down right\n"
        "'x': turn down\n"
        "'z': turn down left\n"
        "'a': turn up left\n"
        "'u': sneak\n"
        "'i': unsneak\n"
        "'j': master weapon attack\n"
        "'k': slave weapon attack\n"
        "Please complete all actions within one frame! \n"
        "[example]: a12sdq2\n"
    ),
    "steps": ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"],
    "gameBeginFlag": False,
}

def refreshUI(ui: UI, packet: PacketResp):
    """Refresh the UI according to the response."""
    global map_info
    global score
    map_info = np.zeros((4, 16, 16))
    data = packet.data
    if packet.type == PacketType.ActionResp:
        ui.playerID = data.playerID
        ui.color = data.color
        ui.characters = data.characters
        ui.score = data.score
        ui.kill = data.kill
        score = data.score

        for block in data.map.blocks:
            if len(block.objs):
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": block.objs[-1].type,
                    "data": block.objs[-1].status,
                }
            else:
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": ObjType.Null,
                }
    subprocess.run(["clear"])
    # ui.display()
    for x in range(mapSize):
        for y in range(mapSize):
            map_info[x][y] = ui.getmap(x, y)

class Client(object):
    """Client obj that send/recv packet.

    Usage:
        >>> with Client() as client: # create a socket according to config file
        >>>     client.connect()     # connect to remote
        >>> 
    """

    def __init__(self) -> None:
        self.config = config
        self.host = self.config.get("Host")
        self.port = self.config.get("Port")
        assert self.host and self.port, "host and port must be provided"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        if self.socket.connect_ex((self.host, self.port)) == 0:
            logger.info(f"connect to {self.host}:{self.port}")
        else:
            logger.error(f"can not connect to {self.host}:{self.port}")
            exit(-1)
        return

    def send(self, req: PacketReq):
        msg = json.dumps(req, cls=JsonEncoder).encode("utf-8")
        length = len(msg)
        self.socket.sendall(length.to_bytes(8, sys.byteorder) + msg)
        # uncomment this will show req packet
        # logger.info(f"send PacketReq, content: {msg}")
        return

    def recv(self):
        length = int.from_bytes(self.socket.recv(8), sys.byteorder)
        result = b''
        while resp := self.socket.recv(length):
            result += resp
            length -= len(resp)
            if length <= 0:
                break

        # uncomment this will show resp packet
        # logger.info(f"recv PacketResp, content: {result}")
        packet = PacketResp().from_json(result)
        return packet

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.socket.close()
        if traceback:
            print(traceback)
            return False
        return True


class player:
    def __init__(self, state_dim, action_dim, save_dir):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # player's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = playerNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Player Net

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
        if self.use_cuda:
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        action_values = self.net(state, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
                Store the experience to self.memory (replay buffer)

                Inputs:
                state (LazyFrame),
                next_state (LazyFrame),
                action (int),
                reward (float),
                done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor(action)
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
                self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss


class playerNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 16:
            raise ValueError(f"Expecting input height: 16, got: {h}")
        if w != 16:
            raise ValueError(f"Expecting input width: 16, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


def cliGetInitReq():
    """Get init request from user input."""
    masterWeaponType = 1# input("Make choices!\nmaster weapon type: [select from {1-2}]: ")
    slaveWeaponType = 1# input("slave weapon type: [select from {1-2}]: ")
    return InitReq(
        MasterWeaponType(int(masterWeaponType)), SlaveWeaponType(int(slaveWeaponType))
    )

def cliGetActionReq(characterID: int):
    """Get action request from user input.

    Args:
        characterID (int): Character's id that do actions.
    """

    def get_action(s: str):
        regex = r"[swedxzauijk]"
        matches = re.finditer(regex, s)
        for match in matches:
            yield match.group()

    str2action = {
        "s": (ActionType.Move, EmptyActionParam()),
        "w": (ActionType.TurnAround, TurnAroundActionParam(Direction.Above)),
        "e": (ActionType.TurnAround, TurnAroundActionParam(Direction.TopRight)),
        "d": (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomRight)),
        "x": (ActionType.TurnAround, TurnAroundActionParam(Direction.Bottom)),
        "z": (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomLeft)),
        "a": (ActionType.TurnAround, TurnAroundActionParam(Direction.TopLeft)),
        "u": (ActionType.Sneaky, EmptyActionParam()),
        "i": (ActionType.UnSneaky, EmptyActionParam()),
        "j": (ActionType.MasterWeaponAttack, EmptyActionParam()),
        "k": (ActionType.SlaveWeaponAttack, EmptyActionParam()),
    }

    actionReqs = []

    actions = input()

    for s in get_action(actions):
        actionReq = ActionReq(characterID, *str2action[s])
        actionReqs.append(actionReq)

    return actionReqs


def recvAndRefresh(ui: UI, client: Client):
    """Recv packet and refresh ui."""
    global gContext
    resp = client.recv()
    refreshUI(ui, resp)

    if resp.type == PacketType.ActionResp:
        if len(resp.data.characters) and not gContext["gameBeginFlag"]:
            gContext["characterID"] = resp.data.characters[-1].characterID
            gContext["playerID"] = resp.data.playerID
            gContext["gameBeginFlag"] = True

    while resp.type != PacketType.GameOver:
        resp = client.recv()
        refreshUI(ui, resp)

    refreshUI(ui, resp)
    print(f"Game Over!")

    for (idx, score) in enumerate(resp.data.scores):
        if gContext["playerID"] == idx:
            print(f"You've got \33[1m{score} score\33[0m")
        else:
            print(f"The other player has got \33[1m{score} score \33[0m")

    if resp.data.result == ResultType.Win:
        print("\33[1mCongratulations! You win! \33[0m")
    elif resp.data.result == ResultType.Tie:
        print("\33[1mEvenly matched opponent \33[0m")
    elif resp.data.result == ResultType.Lose:
        print(
            "\33[1mThe goddess of victory is not on your side this time, but there is still a chance next time!\33[0m"
        )

    gContext["gameOverFlag"] = True
    print("Press any key to exit......")


def main():
    ui = UI()

    with Client() as client:
        client.connect()

        initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
        client.send(initPacket)
        print(gContext["prompt"])

        # IO thread to display UI
        t = Thread(target=recvAndRefresh, args=(ui, client))
        t.start()

        for c in cycle(gContext["steps"]):
            if gContext["gameBeginFlag"]:
                break
            print(
                f"\r\033[0;32m{c}\033[0m \33[1mWaiting for the other player to connect...\033[0m",
                flush=True,
                end="",
            )
            sleep(0.1)

        use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {use_cuda}")
        print()
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        Player = player(state_dim=(16, 16, 4), action_dim=11, save_dir=save_dir)
        logger = MetricLogger(save_dir)
        episodes = 200
        # IO thread accepts user input and sends requests
        global map_info
        global score
        for e in range(episodes):
            while True:

                if gContext["characterID"] is None:
                    continue
                if action := cliGetActionReq(gContext["characterID"]):
                    # Run agent on the state
                    global state_origin
                    state_origin = map_info
                    action = player.act(map_info)
                    actionPacket = PacketReq(PacketType.ActionReq, action)
                    client.send(actionPacket)
                    if state_origin == map_info:
                        continue
                    else:
                        next_state = map_info
                        reward = score
                        done = gContext["gameOverFlag"]
                        Player.cache(state_origin, next_state, action, reward, done)
                        q, loss = Player.learn()
                        # Logging
                        logger.log_step(reward, loss, q)

                        # Update state
                        state_origin = next_state
                if gContext["gameOverFlag"]:
                    break

            logger.log_episode()

            if e % 20 == 0:
                logger.record(episode=e, epsilon=Player.exploration_rate, step=Player.curr_step)
        # gracefully shutdown
        t.join()


if __name__ == "__main__":
    main()
