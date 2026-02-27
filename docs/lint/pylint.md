# PLR* (Pylint Refactor / Design rules)
These are structural / design-smell rules.
Common ones:
Function / Method Size & Shape
PLR0911 — Too many return statements
PLR0912 — Too many branches
PLR0913 — Too many arguments
PLR0914 — Too many local variables
PLR0915 — Too many statements
PLR0916 — Too many boolean expressions
PLR0917 — Too many positional arguments
These are the “this function is doing too much” cluster.
Class Design
PLR0901 — Too many ancestors
PLR0902 — Too many instance attributes
PLR0903 — Too few public methods
PLR0904 — Too many public methods
These catch god-objects and empty shells.
Control Flow / Structure
PLR1702 — Too many nested blocks
PLR1704 — Redefining argument from outer scope
PLR1706 — Consider elif instead of nested if
PLR2004 — Magic value used in comparison

# PLW* — Pylint Warnings (Suspicious Patterns)
These flag code that runs but probably isn’t what you meant.
Common High-Signal Ones
Logic / Flow
PLW0101 — Unreachable code
PLW0120 — Useless else after return
PLW0602 — Global variable not assigned
PLW0603 — Using global
PLW0604 — Using global at module level
Exception Handling
PLW0702 — Bare except
PLW0718 — Broad except Exception
PLW0703 — Catching too general exception
Variable Misuse
PLW0611 — Unused import
PLW0612 — Unused variable
PLW0642 — Self-assignment (x = x)
PLW1501 — Bad file open mode
Comparison Issues
PLW0129 — Assert on constant
PLW0133 — Comparison always true/false

# PLE* — Pylint Errors (Very Likely Bugs)
These usually indicate broken logic or runtime errors.
Core Runtime Errors
Name / Scope
PLE0602 — Undefined variable
PLE0601 — Used before assignment
PLE0118 — Name already defined
Attribute / Call Errors
PLE1101 — No member (attribute doesn’t exist)
PLE1102 — Not callable
PLE1120 — Missing required positional arguments
PLE1121 — Too many positional arguments
PLE1136 — Value not indexable
Control Flow / Syntax
PLE0101 — Return outside function
PLE0102 — Function redefined
PLE1307 — Too many format string args
PLE1205 — Logging format mismatch
