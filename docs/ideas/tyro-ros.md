
### Node pattern

would this make code cleaner or obscure?
dont implement this-

```python
@dataclass
class Config:
    param: str = "default"

class MyNode(Node):
    def __init__(self, cfg: Config | None = None):
        super().__init__("my_node")
        if cfg is None:
            cfg = Config(param=self.declare_parameter("param", "default").value)
        # setup publishers, subscribers, timers

def run(cfg: Config | None = None):
    rclpy.init(); node = MyNode(cfg); rclpy.spin(node)

def main(cfg: Config):
    run(cfg)

if __name__ == "__main__":
    main(tyro.cli(Config))
```

