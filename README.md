# Cp468-A2

# Problem 4

This program models a scheduling problem using a Constraint Satisfaction Problem (CSP).

## Functions

### `plotGraph()`

Draws the constraint graph using matplotlib. Nodes represent variables (X1–X4) and edges represent constraints.

### `satisfies(a, b, constraint)`

Checks if two values satisfy a given constraint (`>=`, `<=`, `>`, `<`, `!=`).

### `AC3()`

Implements the AC-3 algorithm to enforce arc consistency.  
Removes invalid values from domains and returns the updated domains.

### `valid(assignment, graph)`

Checks if the current assignment satisfies all constraints.

### `backtrack(assignment, domains, graph)`

Uses backtracking search to find a valid schedule.

### `main()`

Runs the program by calling AC3 and backtracking, and prints the result.
