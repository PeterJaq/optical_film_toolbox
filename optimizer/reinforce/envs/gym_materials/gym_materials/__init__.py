from gym.envs.registration import register

register(
    id='materials-v0',
    entry_point='gym_materials.envs:MaterialsEnv',
)
# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv',
# )