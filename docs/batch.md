

```python
myspec = {
    "action": {
        'single': (8),
    }
    # deprecated
    # true where action is not padding (along action dim)
    # "action_pad_mask": (8),
    # true where action is used for this head
    "action_head_masks": { "mano": (), "single_arm": (), },
    "observation": {
        "image": {
            "overhead": (224, 224, 3),
            "side": (224, 224, 3),
            "worm": (224, 224, 3),
            "wrist": (224, 224, 3),
        },
        "pad_mask_dict": {
            "image": {
                "overhead": (1,),
                "side": (1,),
                "worm": (1,),
                "wrist": (1,),
            },
            "proprio": {
                "gripper": (1,),
                "joints": (1,),
                "position": (1,),
                "single_arm": (1,),
            },
            "timestep": (1,),
        },
        "proprio": {
            "gripper": (1),
            "joints": (7),
            "position": (6),
            "single_arm": (14),
        },
        # marks if task / goal is after the state observation ???
        "task_completed": (1, 50),
        "timestep": (1,),
        # (b, w) Boolean mask that is False when the state timestep corresponds to padding
        "timestep_pad_mask": (1,),
    },
    # goal conditioning tokens
    "task": {
        "image": {
            "overhead": (224, 224, 3),
            "side": (224, 224, 3),
            "worm": (224, 224, 3),
            "wrist": (224, 224, 3),
        },
        "language.embedding": (512,),
        "pad_mask_dict": {
            "image": {
                "overhead": (),
                "side": (),
                "worm": (),
                "wrist": (),
            },
            "language.embedding": (),
            "proprio": {
                "gripper": (),
                "joints": (),
                "position": (),
                "single_arm": (),
            },
            "timestep": (),
        },
        "proprio": {
            "gripper": (1,),
            "joints": (7,),
            "position": (6,),
            "single_arm": (14,),
        },
        "timestep": (),
    },
}
```

