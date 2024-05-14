from typing import Dict, Any
from fsrl.trainer import OffpolicyTrainer

class OffpolicyTrainer(OffpolicyTrainer):

    def test_step(self) -> Dict[str, Any]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        self.test_collector.reset_env()
        self.test_collector.reset_buffer()
        self.policy.eval()
        stats_test = self.test_collector.collect(n_episode=self.episode_per_test)

        self.logger.store(
            **{
                "test/reward": stats_test["rew"],
                "test/cost": stats_test["cost"],
                "test/length": int(stats_test["len"]),
                "test/feasible_reward": stats_test["feasible_rew"],
                "test/violation_rate": stats_test["violation_rate"]
            }
        )
        return stats_test

    def train_step(self) -> Dict[str, Any]:
        """Perform one training step."""
        assert self.episode_per_test is not None
        stats_train = self.train_collector.collect(self.episode_per_collect)

        self.env_step += int(stats_train["n/st"])
        self.cum_cost += stats_train["total_cost"]
        self.cum_episode += int(stats_train["n/ep"])
        self.logger.store(
            **{
                "update/episode": self.cum_episode,
                "update/cum_cost": self.cum_cost,
                "train/reward": stats_train["rew"],
                "train/cost": stats_train["cost"],
                "train/length": int(stats_train["len"]),
                "train/feasible_reward": stats_train["feasible_rew"],
                "train/violation_rate": stats_train["violation_rate"]
            }
        )
        return stats_train
    
