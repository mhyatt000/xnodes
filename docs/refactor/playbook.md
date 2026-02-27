# Refactoring Playbook

A distilled reference for safe, sustainable refactoring of large software systems.
Synthesised from Martin Fowler's *Refactoring* (2nd ed.), *Refactoring at Scale* (Maude Lemaire),
and current engineering practice as of 2025.

---

## 1. Principles

### 1.1 What refactoring is (and is not)

Refactoring means restructuring existing code to improve its internal structure
**without changing its externally observable behaviour**.
It is not: adding features, fixing bugs, or optimising performance.
Mixing concerns in a single commit makes it impossible to review or bisect later.

### 1.2 Core design heuristics

| Acronym | Rule | Practical effect |
|---------|------|-----------------|
| **SRP** | A module has one reason to change | Smaller, focused classes / functions |
| **OCP** | Open for extension, closed for modification | Add via new code, not edits to stable code |
| **DIP** | Depend on abstractions, not concretions | Inject dependencies; invert control |
| **DRY** | Don't repeat yourself | Extract shared logic to one canonical location |
| **YAGNI** | You ain't gonna need it | Delete speculative code; add only when required |
| **KISS** | Keep it simple | Prefer the plain solution; complexity is a cost |

These heuristics are filters, not laws.
Apply the one most relevant to the smell in front of you.

### 1.3 Complexity budget

Measure complexity per function, not per file:

- **Cyclomatic complexity ≤ 10** (each branch or loop +1)
- **Cognitive complexity ≤ 10** (nesting adds more weight than branching)
- **Maintainability Index > 20** (composite of LOC, cyclomatic, Halstead)

This project already targets cyclomatic complexity ≤ 8 (prefer 4–5) per `CLAUDE.md`.

---

## 2. Before You Touch Anything

### 2.1 Understand the blast radius

Read the code. Trace call sites. Identify what tests exist.
Do not refactor code you have not read.

### 2.2 Write tests first

If the unit under refactor lacks tests, write characterisation tests before changing anything.
These are not quality tests — they pin the *current* behaviour, however imperfect, so regressions surface immediately.

```
Checklist before first commit
  ☐ All existing tests pass
  ☐ New tests cover the paths you will touch
  ☐ CI is green on the current commit
```

### 2.3 Define "done"

Large refactors fail when the finish line is vague.
Write down the target state: *what does the system look like when the refactor is complete?*
Aim for "better than today", not "perfect in theory".
Map the shortest path — every step should be independently shippable.

---

## 3. Refactoring Safely: The Core Loop

```
Red → Green → Refactor → Commit → Repeat
```

1. **Make one small, behaviour-preserving change.**
2. Run the tests.
3. If green: commit with an atomic message (e.g. `refactor: extract validate_pose() from anchor node`).
4. If red: revert the change, understand why, try again.

Never accumulate multiple refactorings in an uncommitted state.
Small commits make bisection trivial and reviewers happy.

---

## 4. Catalogue of Common Refactoring Moves

### 4.1 Composing methods

| Move | When to apply |
|------|--------------|
| **Extract Method** | A code block has a clear purpose separable from its host; comments explain *what* it does |
| **Inline Method** | A method body is as clear as its name; the indirection adds no value |
| **Replace Temp with Query** | A local variable is computed once and read several times; make it a property/function |
| **Replace Method with Method Object** | Extraction is blocked by too many shared locals; make the function a class |
| **Substitute Algorithm** | A clearer algorithm exists for the same result |

### 4.2 Moving features between objects

| Move | When to apply |
|------|--------------|
| **Move Method / Field** | A method uses another class's data more than its own |
| **Extract Class** | A class has grown two distinct sets of responsibilities |
| **Hide Delegate** | Client talks to an intermediary's internals; expose only what is needed |

### 4.3 Simplifying conditionals

| Move | When to apply |
|------|--------------|
| **Decompose Conditional** | Branches are complex; extract condition and bodies into named functions |
| **Replace Nested Conditional with Guard Clauses** | Normal path is buried in else-clauses; return early on exceptional paths |
| **Replace Conditional with Polymorphism** | A switch/if-elif dispatches on type; create subclasses that override one method |
| **Introduce Null Object** | Code littered with `if x is None` checks; provide a do-nothing default object |

### 4.4 Dealing with generalisation

| Move | When to apply |
|------|--------------|
| **Pull Up Method / Field** | Identical logic in sibling classes; move to parent |
| **Extract Interface** | Callers need only a subset of a class; expose only that contract |
| **Replace Inheritance with Delegation** | Subclass uses only part of superclass; compose instead |

---

## 5. Addressing Code Smells

| Smell | Likely refactoring |
|-------|--------------------|
| Duplicated code | Extract Method → pull up or delegate |
| Long method | Extract Method, Replace Temp with Query |
| Large class | Extract Class, Extract Subclass |
| Long parameter list | Introduce Parameter Object, Replace with Config dataclass |
| Divergent change (one class changes for many reasons) | Extract Class |
| Shotgun surgery (one change touches many classes) | Move Method/Field, Inline Class |
| Feature envy (method uses another class more than its own) | Move Method |
| Data clumps (same group of fields everywhere) | Extract Class, Introduce Parameter Object |
| Primitive obsession | Replace Primitive with Object (e.g. use `pathlib.Path`, not `str`) |
| Switch / if-elif on type | Replace Conditional with Polymorphism |
| Speculative generality (unused abstractions) | Inline Class, Remove Middle Man |
| Comments explaining *what* | Extract Method; let the name be the comment |

---

## 6. Decoupling Strategies for Large Systems

### 6.1 Find seams

A *seam* is a place where code can be changed without editing that place directly.
Look for natural seams where:
- data flow between subsystems is minimal,
- business logic is cohesive within a candidate boundary,
- team ownership aligns with the split.

### 6.2 Introduce a Facade

When a subsystem is heavily entangled with callers, wrap it behind a Facade with
a stable interface. Callers see only the Facade; the internals can be rearranged freely.

### 6.3 Branching by Abstraction

1. Create an abstraction (interface / abstract class) that wraps the component to change.
2. Make all callers use the abstraction.
3. Implement the new behaviour behind the abstraction.
4. Switch callers to the new implementation incrementally.
5. Delete the old implementation.

No feature flags; no long-lived forks.

### 6.4 Strangler Fig Pattern

For large-scale migration (monolith → modular / microservices):

1. Identify the subsystem to replace.
2. Introduce an indirection layer (proxy, API gateway, or dispatch table).
3. Implement replacement code behind the indirection layer.
4. Migrate callers one slice at a time, with tests green after each migration.
5. Delete the legacy code once fully replaced.

The legacy system keeps running throughout; risk is contained to each slice.

### 6.5 Event-driven decoupling

Replace direct calls with events when source and target should remain independent.
The emitter publishes an event; it does not know (or care) who consumes it.
Reduces coupling at the cost of traceability — use with deliberate observability.

---

## 7. Managing Technical Debt

### 7.1 Measure before you prioritise

Collect baseline metrics before starting any large refactor:

```bash
# Example: ruff for Python complexity
ruff check --select C901 src/          # cyclomatic complexity violations
ruff check --select PLR0912 src/       # too many branches
```

Integrate thresholds in CI so debt does not silently accumulate.

### 7.2 Prioritise by impact

Not all debt is equal. Rank by:

1. **Churn × complexity** — files that change often *and* are hard to understand are the highest priority.
2. **Defect density** — modules with the most bug reports.
3. **Business criticality** — hot paths in production.

Teams that prioritise high-impact components see ~4× better ROI than those doing
comprehensive rewrites uniformly.

### 7.3 The Boy Scout Rule

Leave every module slightly better than you found it.
Fix one smell per PR, not ten. Momentum beats heroic sprints.

### 7.4 Don't drop a large refactor half-way

A partially refactored codebase is often *harder* to work with than the original.
Ensure the team has the bandwidth to carry the work to the defined finish line
before starting.

---

## 8. Process & Team

### 8.1 Make refactoring a continuous practice

Organisations that embed refactoring into the regular development cycle outperform
those who treat it as periodic big-bang projects.

- Allocate capacity per sprint (e.g. 10–20% of story points).
- Pair refactoring with the feature work that touches the same files.
- Track the Tech Debt Ratio (remediation cost ÷ rewrite cost); set an acceptable ceiling.

### 8.2 Code review culture

Code review is the primary mechanism for catching regressions *and* for knowledge transfer.
Reviewers should verify:

- External behaviour is unchanged (tests prove this).
- Naming is clearer after the refactor.
- No new coupling was introduced.
- Complexity metrics did not increase.

Use metrics as conversation starters, not weapons.

### 8.3 CI/CD as a maintainability gate

Run static analysis on every PR:

```yaml
# Typical CI checks for maintainability
- ruff check (lint + complexity)
- ruff format --check
- pytest (unit + integration coverage threshold)
- pre-commit hooks
```

Set hard failure thresholds for complexity and coverage regressions;
treat them like compilation errors.

### 8.4 Team structure

Large refactors need:
- A small core group with full-time ownership.
- Domain experts available for async consultation.
- Documentation of the process so other teams can self-serve.

When refactoring crosses team boundaries, help others do the work
(write codemods, provide tests, pair) rather than doing it all yourself.

---

## 9. Quick Reference: Refactor Decision Tree

```
Is the behaviour tested?
  No → Write characterisation tests first
  Yes ↓

Is the change small enough to complete in one sitting?
  No → Break into independent slices; define done for each
  Yes ↓

Does the change preserve the public interface?
  No → Use Facade / Branching by Abstraction to maintain backward compat
  Yes ↓

Commit → CI green → Ship → Repeat
```

---

## Sources

- [Code Refactoring in 2025: Best Practices & Popular Techniques](https://marutitech.medium.com/best-practices-code-refactoring-19c81263ac43)
- [How to Refactor Complex Codebases – freeCodeCamp](https://www.freecodecamp.org/news/how-to-refactor-complex-codebases/)
- [Key points of Refactoring at Scale](https://understandlegacycode.com/blog/key-points-of-refactoring-at-scale/)
- [The Future of Software Maintainability – Qodo](https://www.qodo.ai/blog/software-maintainability/)
- [Code Maintainability Best Practices – Number Analytics](https://www.numberanalytics.com/blog/ultimate-guide-code-maintainability)
- [Maintainability in Software – Tricentis](https://www.tricentis.com/learn/maintainability)
- [Strangler Fig Pattern – Thoughtworks](https://www.thoughtworks.com/en-us/insights/articles/embracing-strangler-fig-pattern-legacy-modernization-part-one)
- [Strangler Fig Application Pattern – microservices.io](https://microservices.io/post/refactoring/2023/06/21/strangler-fig-application-pattern-incremental-modernization-to-services.md.html)
- [Refactoring Catalog – refactoring.com](https://refactoring.com/catalog/)
- [Replace Conditional with Polymorphism – refactoring.guru](https://refactoring.guru/replace-conditional-with-polymorphism)
- [Extract Method – refactoring.guru](https://refactoring.guru/extract-method)
- [Understanding Code Complexity – Qodo](https://www.qodo.ai/blog/code-complexity/)
- [Maintainability Metrics – SonarQube Docs](https://docs.sonarsource.com/sonarqube-server/user-guide/code-metrics/metrics-definition)
- [Our 10-Point Technical Debt Assessment – Code Climate](https://codeclimate.com/blog/10-point-technical-debt-assessment)
- [Decoupling Dependencies in Software Architecture – LinkedIn](https://www.linkedin.com/pulse/decoupling-dependencies-software-architecture-path-scalable-childs-vl1ne)
- [Microservices Architecture and Legacy System Decomposition – SoftwareSeni](https://www.softwareseni.com/microservices-architecture-and-legacy-system-decomposition-strategies/)
- [AI Code Refactoring – GetDX](https://getdx.com/blog/enterprise-ai-refactoring-best-practices/)
