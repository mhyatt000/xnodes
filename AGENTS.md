# THINGS I LIKE

* I prefer very concise docstrings. and concise code. the purpose of the code
  should be clear from reading its contents

if its a dataclass, then
@dataclass class Myclass:
""" Myclass is a dataclass that represents a simple example. """
    name: str # it is better to describe the argument here
    age: int # and here rather than the docstring

* I prefer OOP paterns for different components.
* wrapper/decorator can sometimes be a good choice
* I also like using a config.create(*args, **kwargs) to create components from
  config objects

* don't be afraid to propose new code restructuring and reorganizing if doing
  so benefits maintainability, but do not do so frivolously

**multi file organization**
*	Use feature-based style rather than layer based. Keeps complexity isolated
*	Add a shared/infra or common/ layer for reusable low-level primitives

### The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!


# THINGS I DON'T LIKE

* excessive nesting. except when necessary, components should be flat. list and
  dict comprehension is fine. jax.tree.map is also fine
* long variable names. this clutters the code. try to keep them short and
  meaningful.
* Long Functions or Methods. multiple responsibilities can be challenging to
  test and maintain. Break down long functions into smaller, single-purpose
  functions to adhere to the Single Responsibility Principle. cyclomatic complexity (CC) should be no more than 8, but 4-5 is better.
* Duplicated Code. keep it dry
