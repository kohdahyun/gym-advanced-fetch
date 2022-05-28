from gym.envs.registration import register

register(
        id="FetchReachAndThrow-v0",
        entry_point="gym_advanced_fetch.envs:FetchReachAndThrowEnv",
        max_episode_steps=50,
        )

register(
        id="FetchReachWithObstacle-v0",
        entry_point="gym_advanced_fetch.envs:FetchReachWithObstacleEnv",
        max_episode_steps=50,
        )

register(
        id="FetchReachAndThrowWithPath-v0",
        entry_point="gym_advanced_fetch.envs:FetchReachAndThrowWithPathEnv",
        max_episode_steps=50,
        )

register(
        id="FetchReachAndThrowFix-v0",
        entry_point="gym_advanced_fetch.envs:FetchReachAndThrowFixEnv",
        max_episode_steps=50,
        )

register(
        id="FetchReachAndThrowFixCgbr-v0",
        entry_point="gym_advanced_fetch.envs:FetchReachAndThrowFixCgbrEnv",
        max_episode_steps=50,
        )

register(
        id="FetchReachAndThrowFixCgbrZero-v0",
        entry_point="gym_advanced_fetch.envs:FetchReachAndThrowFixCgbrZeroEnv",
        max_episode_steps=50,
        )