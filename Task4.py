import matplotlib.pyplot as plt
from collections import deque

'''
CP468 Assignment – Constraint Satisfaction Problem

This program implements the AC-3 algorithm to enforce arc consistency
for an employee shift scheduling problem. After reducing the domains,
a backtracking search is used to find a valid assignment of shifts
that satisfies all constraints.

Approach:
This program models the scheduling task as a Constraint
Satisfaction Problem (CSP) and solves it in two main stages:

1. Arc Consistency (AC-3 Algorithm)
   The AC-3 algorithm is applied to remove values from domains
   that violate constraints. This reduces the search space by
   ensuring arc consistency between variables.

2. Backtracking Search
   After domain reduction, a backtracking search algorithm
   assigns specific shift values to each employee while ensuring
   that all constraints remain satisfied.

The program also includes an optional function to visualize the
constraint graph where variables are nodes and constraints are
represented as edges.'''

# Draw the constraint graph
def plotGraph():

    # positions of each node
    positions = {
        "X1": (0,0),
        "X2": (1,0),
        "X3": (2,0),
        "X4": (3,0)
    }

    # define the edges and constraints from one node to another
    edges = [
        ("X1","X2","≥"),
        ("X2","X3",">"),
        ("X3","X4","≠")
    ]

    # draw nodes
    for node,(x,y) in positions.items():
        plt.scatter(x,y,s=2000)
        plt.text(x,y,node,ha='center',va='center',color='white')

    # draw edges
    for start,end,label in edges:

        x1,y1 = positions[start]
        x2,y2 = positions[end]

        plt.plot([x1,x2],[y1,y2])

        midx = (x1+x2)/2
        midy = (y1+y2)/2

        plt.text(midx,midy+0.05,label,ha='center')

    plt.axis("off")

    #show the graph
    plt.show()


# Check constraints
def satisfies(a,b,constraint):

    if constraint == ">=":
        return a >= b

    if constraint == "<=":
        return a <= b

    if constraint == ">":
        return a > b

    if constraint == "<":
        return a < b

    if constraint == "!=":
        return a != b


# Implement AC-3 algorithmn
def AC3():

    # domains
    domains = {
        "X1":[1,2,3,4],
        "X2":[3,4,5,8,9],
        "X3":[2,3,5,6,7,9],
        "X4":[3,5,7,8,9]
    }

    # constraint graph
    graph = {
        "X1":{"X2":">="},
        "X2":{"X1":"<=","X3":">"},
        "X3":{"X2":"<","X4":"!="},
        "X4":{"X3":"!="}
    }

    # initial queue
    queue = deque([
        ("X1","X2"),
        ("X2","X1"),
        ("X2","X3"),
        ("X3","X2"),
        ("X3","X4"),
        ("X4","X3")
    ])

    while queue:

        
        Xi,Xj = queue.popleft()
        constraint = graph[Xi][Xj]

        revised = False

        # check each value in Xi
        for x in domains[Xi][:]:

            # checks if value is valid in a list or not according to the defined constraint
            valid = False

            for y in domains[Xj]:

                # Checks if x,y satisfies the constraint
                if satisfies(x,y,constraint):
                    valid = True
                    break

            if not valid:
                domains[Xi].remove(x)
                revised = True

        # if domain changed add neighbors back to queue
        if revised:

            if len(domains[Xi]) == 0:
                print("No solution exists")
                return

            for Xk in graph[Xi]:
                if Xk != Xj:
                    queue.append((Xk,Xi))

    print("Final Domains:")
    for var in domains:
        print(var,"=",domains[var])
    
    return domains

def valid(assignment, graph):

    # Check constraints between assigned variables
    for Xi in assignment:
        for Xj in graph.get(Xi, {}):

            if Xj in assignment:

                a = assignment[Xi]
                b = assignment[Xj]
                constraint = graph[Xi][Xj]

                if constraint == ">=" and not (a >= b):
                    return False
                if constraint == "<=" and not (a <= b):
                    return False
                if constraint == ">" and not (a > b):
                    return False
                if constraint == "<" and not (a < b):
                    return False
                if constraint == "!=" and not (a != b):
                    return False

    return True

# Backtracking search to find a complete assigned variables for X1, X2, X3, X4
def backtrack(assignment, domains, graph):

    # If all variables assigned, solution found
    if len(assignment) == len(domains):
        return assignment

    # Select an unassigned variable
    for var in domains:
        if var not in assignment:
            Xi = var
            break

    # Try values from the variable's domain
    for value in domains[Xi]:

        assignment[Xi] = value

        # Check if assignment remains valid
        if valid(assignment, graph):

            result = backtrack(assignment, domains, graph)

            if result:
                return result
        
         # Undo assignment if it fails
        del assignment[Xi]

    return None

def main():

    # close the graph to see the results on terminal
    # plot the constraint graph
    plotGraph()

    # Run AC-3 to get reduced domains
    domains = AC3()

    print("\nDomains after AC-3:")
    for v in domains:
        print(v, "=", domains[v])

    # Constraint graph used for search
    graph = {
        "X1":{"X2":">="},
        "X2":{"X1":"<=","X3":">"},
        "X3":{"X2":"<","X4":"!="},
        "X4":{"X3":"!="}
    }

    # Run backtracking search to find a valid schedule
    solution = backtrack({}, domains, graph)

    print("\nOne Valid Schedule:")
    print(solution)

# Run the program
if __name__ == "__main__":
    main()

