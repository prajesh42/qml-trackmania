# standard library imports
import os
import pickle

# third-party imports
import numpy as np
import logging


class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """

    def __init__(
        self,
        reward_data_path,
        nb_obs_forward=10,
        nb_obs_backward=10,
        nb_zero_rew_before_failure=30,
        min_nb_steps_before_failure=int(3.5 * 20),
        max_dist_from_traj=60.0,
    ):
        """
        Instantiates a reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts
            nb_obs_backward: rewind distance in trajectory
            nb_zero_rew_before_failure:
                number of steps with no progress before termination
            min_nb_steps_before_failure:
                minimum episode length before failure logic activates
            max_dist_from_traj:
                maximum allowed distance from demo trajectory
        """

        if not os.path.exists(reward_data_path):
            logging.error(
                f"Reward data not found at path: {reward_data_path}. Using dummy reward."
            )
            self.data = np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            )
        else:
            with open(reward_data_path, "rb") as f:
                try:
                    self.data = pickle.load(f)
                except Exception as e:
                    logging.error(
                        f"Failed to load reward data: {e}. Using dummy reward."
                    )
                    self.data = np.array(
                        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
                    )

        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = max(
            5,
            nb_zero_rew_before_failure,
        )
        self.min_nb_steps_before_failure = (
            min_nb_steps_before_failure
        )
        self.max_dist_from_traj = (
            max_dist_from_traj
            if max_dist_from_traj > 0
            else 100.0
        )

        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)

        # used for steering smoothness reward
        self.prev_steer = 0.0

        # used for speed impact penalty
        self.prev_speed = 0.0

    def compute_reward(
        self,
        pos,
        steer=0.0,
        speed=0.0,
        collision=False,
    ):
        """
        Computes reward.

        Args:
            pos:
                current car position

            steer:
                current steering action
                expected range: [-1, 1]

            speed:
                current speed

            collision:
                bool indicating wall collision

        Returns:
            reward, terminated
        """

        terminated = False

        self.step_counter += 1

        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = self.cur_idx

        # forward search in trajectory
        while True:

            dist = np.linalg.norm(
                pos - self.data[index]
            )

            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward

            index += 1
            temp -= 1

            if (
                index >= self.datalen
                or temp <= 0
            ):

                # hard cutoff if extremely far
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx

                break

        # ==========================================================
        # BASE PROGRESS REWARD
        # ==========================================================

        progress = (
            best_index - self.cur_idx
        ) / 100.0

        reward = progress

        # ==========================================================
        # CENTERLINE / TRAJECTORY DISTANCE PENALTY
        # ==========================================================

        # smooth continuous penalty
        reward -= 0.002 * min_dist

        # slightly stronger nonlinear penalty
        reward -= 0.0005 * (min_dist ** 1.2)

        # ==========================================================
        # STEERING SMOOTHNESS PENALTY
        # ==========================================================

        steer_change = abs(
            steer - self.prev_steer
        )

        reward -= 0.01 * steer_change

        self.prev_steer = steer

        # ==========================================================
        # SPEED DROP PENALTY
        # ==========================================================

        speed_drop = max(
            0.0,
            self.prev_speed - speed,
        )

        reward -= 0.002 * speed_drop

        self.prev_speed = speed

        # ==========================================================
        # COLLISION PENALTY
        # ==========================================================

        if collision:
            reward -= 1.5

        # ==========================================================
        # REWIND LOGIC
        # ==========================================================

        if best_index == self.cur_idx:

            min_dist = np.inf
            index = self.cur_idx

            while True:

                dist = np.linalg.norm(
                    pos - self.data[index]
                )

                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward

                index -= 1
                temp -= 1

                if (
                    index <= 0
                    or temp <= 0
                ):
                    break

            # failure logic
            if (
                self.step_counter
                > self.min_nb_steps_before_failure
            ):

                self.failure_counter += 1

                if (
                    self.failure_counter
                    > self.nb_zero_rew_before_failure
                ):
                    terminated = True

        else:
            self.failure_counter = 0

        # update trajectory index
        self.cur_idx = best_index

        return reward, terminated

    def reset(self):
        """
        Resets reward function.
        """

        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0

        self.prev_steer = 0.0
        self.prev_speed = 0.0